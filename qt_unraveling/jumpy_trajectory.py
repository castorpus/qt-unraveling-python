'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, objmode, float64, complex128, int64

import qt_unraveling.usual_operators as op
import qt_unraveling.misc_func as misc

# Import integrators
from qt_unraveling.integrators import scipy_integrator, schrodinger_operator

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def jumpTrajectory_(initialState, timelist, drivingH,
                       original_lindbladList, seed=10):
    # Resetting the random generator
    np.random.seed(seed)
    # Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]

    # Jump operators
    jump_op_list = original_lindbladList
    
    # Jump probability
    def jumpProb(state, J_op_list):
        """ When the system jumps, this function determines which jump 
        the system makes. 
        """

        weight = np.zeros(np.shape(J_op_list)[0], dtype=np.float64)
        J_idx = np.zeros(np.shape(J_op_list)[0], dtype=np.int64)
        for mu, Jmu in enumerate(J_op_list):
            Jmu = np.ascontiguousarray(Jmu)
            Jmu_dag_Jmu = np.dot(np.conjugate(np.transpose(Jmu)), Jmu)
            weight[mu] += np.real(np.vdot(state,np.dot(Jmu_dag_Jmu, state)))
            J_idx[mu] += mu 
        
        j_prob = sum(weight)
        ## This case should be consider
        # if j_prob == 0:
        
        weight_renorm = np.cumsum(weight/j_prob)
        jump_dice = np.random.rand()
        j_idx = np.searchsorted(weight_renorm, jump_dice)
        
        return j_idx

    ## The no-jump part for the integrator
    #
    # no-jump operator
    no_jump_op = drivingH - (1/2)*1j*sum(
        [np.dot(np.conjugate(np.transpose(x)), x) for x in jump_op_list]
        )
    # Making the operator time dependent.   
    def hamiltonian_eff(t, H_eff = no_jump_op): 
        return H_eff
    # Defining the generator for the integrator
    no_jump_gen = lambda psi_it, it: schrodinger_operator(hamiltonian_eff, psi_it, it) 

    ## Jump-trajectory evolution
    # Fixing the memory space
    psi_trajectory = np.ascontiguousarray(np.zeros(np.shape(timelist) 
                                                   + np.shape(initialState), dtype=np.complex128))
    # Initializing the trajectory
    psi_trajectory[0] += initialState # Setting the initial state
    jump_prob = np.random.rand()

    for n_it in np.arange(timeSteps-1):
        state_step = psi_trajectory[n_it]
        state_step_normSq = np.linalg.norm(state_step)**2

        if state_step_normSq > jump_prob:
            state_step = scipy_integrator(no_jump_gen, 
                                                  state_step, np.array([0.0, dt]), method='BDF', rrtol=1e-5, aatol=1e-5, last_point=True)[0]
            psi_trajectory[n_it+1] += state_step
        else:
            jump_op_idx = jumpProb(state_step, jump_op_list)
            state_step = np.dot(jump_op_list[jump_op_idx], state_step)
            state_step_normSq = np.linalg.norm(state_step)**2
            psi_trajectory[n_it+1] += state_step/np.linalg.norm(state_step)
            jump_prob = np.random.rand()

    psi_trajectory_norm = [state/np.linalg.norm(state) 
                            for state in psi_trajectory]

    return psi_trajectory_norm


#@njit(int64(complex128[:], complex128[:,:,:], float64, int64))
def dNRho(stateRho, measurement_op_list, dt, seed=0):
    weight = np.zeros(np.shape(measurement_op_list)[0], dtype=np.float64)
    M_index = np.zeros(np.shape(measurement_op_list)[0], dtype=np.int64)

    for mu, Mmu in enumerate(measurement_op_list):
        Mmu = np.ascontiguousarray(Mmu)
        Mmu_dag_Mmu = np.dot(np.conjugate(np.transpose(Mmu)), Mmu)
        weight[mu] += dt*np.real(np.trace(np.dot(stateRho, Mmu_dag_Mmu)))
        M_index[mu] += mu
    
    np.random.seed(seed)
    R = np.random.rand()
    if R < sum(weight): 
        jump_index = misc.numba_choice(M_index, weight, 1)[0]
    else:
        jump_index = np.shape(measurement_op_list)[0]
    return jump_index

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def ortogonal_mixing(oMatrix, coherent_fields, L_it):
    new_ops = np.zeros(np.shape(L_it), dtype=np.complex128)
    for n_O, O in enumerate(oMatrix):
        new_ops[n_O] += coherent_fields[n_O]*np.eye(np.shape(L_it)[0])
        for n_L, L in enumerate(L_it):
            new_ops[n_O] += O[n_L]*L
    return new_ops

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def coherent_field_mixing(coherent_fields, L_it):
    new_ops = np.zeros(np.shape(L_it), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        new_ops[n_L] += L + coherent_fields[n_L]*np.eye(np.shape(L)[0])
    return new_ops

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def jumpRhoTrajectory_td(initialStateRho, timelist, drivingH, original_lindbladList, 
                         eta_diag, lindbladList, coherent_fields, seed):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]

    ## Number of Lindblad operators
    num_lindblad_channels = np.shape(lindbladList(timelist[0],initialStateRho))[0]

    rho_trajectory = np.ascontiguousarray(np.zeros(np.shape(timelist) + np.shape(initialStateRho), dtype=np.complex128))
    rho_trajectory[0] += initialStateRho

    no_jump_term_1 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    no_jump_term_2 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    inefficient_term = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))

    for n_it, it in enumerate(timelist[:-1]):
        L_it = lindbladList(it, rho_trajectory[n_it])
        original_L_it = original_lindbladList(it)

        for n_i, L_i in enumerate(original_L_it):
            inefficient_term += (1 - eta_diag[n_i])*op.D(L_i, rho_trajectory[n_it])*dt

        ## coherent_field mix
        #ortogonal_ops = ortogonal_mixing(oMatrix, coherent_fields, L_it)
        coherent_field_ops = coherent_field_mixing(coherent_fields, L_it)

        ## Jump index
        jump_index = dNRho(rho_trajectory[n_it], coherent_field_ops, dt, seed*timeSteps+n_it)

        ## Euler step
        if jump_index < num_lindblad_channels:
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.G(coherent_field_ops[jump_index], rho_trajectory[n_it]) + inefficient_term

        elif jump_index == num_lindblad_channels:
            for n_r, L_r in enumerate(L_it):
                L_r = np.ascontiguousarray(L_r)
                no_jump_term_1 += -0.5*np.dot(np.transpose(np.conjugate(L_r)), L_r)
                no_jump_term_2 += -coherent_fields[n_r]*L_r
            
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.H(-1j*drivingH(it) + no_jump_term_1 + no_jump_term_2, rho_trajectory[n_it])*dt + inefficient_term
            no_jump_term_1, no_jump_term_2, inefficient_term = 0*no_jump_term_1, 0*no_jump_term_2, 0*inefficient_term

    return rho_trajectory

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def jumpRhoTrajectory_(initialStateRho, timelist, drivingH,
                       original_lindbladList, eta_diag, lindbladList, coherent_fields, seed):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]

    ## Number of Lindblad operators
    num_lindblad_channels = np.shape(lindbladList)[0]

    rho_trajectory = np.ascontiguousarray(np.zeros(np.shape(timelist) 
                                                   + np.shape(initialStateRho), 
                                                   dtype=np.complex128))
    rho_trajectory[0] += initialStateRho

    no_jump_term_1 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    no_jump_term_2 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    inefficient_term = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    for n_it, it in enumerate(timelist[:-1]):
        L_it = lindbladList
        original_L_it = original_lindbladList

        for n_i, L_i in enumerate(original_L_it):
            inefficient_term += (1 - eta_diag[n_i])*op.D(L_i, rho_trajectory[n_it])*dt

        ## coherent_field mix
        #ortogonal_ops = ortogonal_mixing(oMatrix, coherent_fields, L_it)
        coherent_field_ops = coherent_field_mixing(coherent_fields, L_it)

        ## Jump index
        jump_index = dNRho(rho_trajectory[n_it], coherent_field_ops, dt, seed*timeSteps+n_it)

        ## Euler step
        if jump_index < num_lindblad_channels:
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.G(coherent_field_ops[jump_index], rho_trajectory[n_it]) + inefficient_term

        elif jump_index == num_lindblad_channels:
            for n_r, L_r in enumerate(L_it):
                L_r = np.ascontiguousarray(L_r)
                no_jump_term_1 += -0.5*np.dot(np.transpose(np.conjugate(L_r)), L_r)
                no_jump_term_2 += -coherent_fields[n_r]*L_r
            
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.H(-1j*drivingH + no_jump_term_1 + no_jump_term_2, rho_trajectory[n_it])*dt + inefficient_term
            no_jump_term_1, no_jump_term_2, inefficient_term = 0*no_jump_term_1, 0*no_jump_term_2, 0*inefficient_term

    return rho_trajectory