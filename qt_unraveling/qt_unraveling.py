"""
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
"""

import numpy as np
from scipy.linalg import sqrtm, polar
from numba import objmode
import warnings
from functools import partial

# Import parallel auxiliary function
from qt_unraveling.misc_func import parallel_run

# Import numba optimized functions
from qt_unraveling.usual_operators import operators_Mrep, sqrt_jit, PsiToRho

# Trajectory modules
from qt_unraveling.diffusive_trajectory import diffusiveRhoTrajectory_, diffusiveRhoTrajectory_td
from qt_unraveling.feedback_trajectory import feedbackRhoTrajectory_, feedbackRhoTrajectory_delay
from qt_unraveling.jumpy_trajectory import jumpTrajectory_,jumpRhoTrajectory_, jumpRhoTrajectory_td

# Import integrators
from qt_unraveling.integrators import custom_rungekutta_integrator, scipy_integrator, schrodinger_operator, vonneumann_operator, GKSL_operator, feedbackEvol_operator

class QuantumSystem:
    def __init__(self, drivingH, initialState, timeList, *, J_list=None,
                 TMatrix=None, c_fields=None):
        """
        Initialize the system with the provided parameters.
        """ 
        self.setup_time_interval(timeList)
        self.setup_initial_state(initialState)
        self.setup_hamiltonian(drivingH)

        if J_list is not None:
            self.setup_unraveling(J_list, TMatrix, c_fields)     
            self.setup_jumpy_methods()  
    
    def setup_time_interval(self, timeList):
        self.timeList = timeList
        self.t0 = timeList[0]
        self.tmax = timeList[-1]
        self.maxiter = np.shape(timeList)[0]
        self.dt = abs(timeList[1] - timeList[0])

    def setup_initial_state(self, initialState):
        self.dimH = np.shape(initialState)[0]
        if len(np.shape(initialState)) == 1:  # for pure states
            self.initial_state_type = 0
            if np.round(np.linalg.norm(initialState), 5) != 1:
                warnings.warn('Initial state is unnormalized. Normalized state taken instead')
                self.initialStatePsi = (1. / np.linalg.norm(initialState)) * initialState
            else:
                self.initialStatePsi = initialState
            self.initialStateRho = np.asarray(np.transpose(np.asmatrix(self.initialStatePsi)).dot(np.conjugate(np.asmatrix(self.initialStatePsi))))  # provides the correspondent density matrix
        elif len(np.shape(initialState)) == 2:
            self.initial_state_type = 1  # for mixed states
            self.initialStateRho = np.asarray(initialState)
            """
            For future work: 
            Note that to evolve mixed states an eigendecomposition into pure
            states is necessary.
            """

    def setup_hamiltonian(self, drivingH):
        if not (type(drivingH).__name__ in ['ndarray', 'CPUDispatcher']):
            raise ValueError('System Hamiltonian must be passed as a jitted function or ndarray of dtype complex128')

        self.timedepent_hamiltonian = False
        if type(drivingH).__name__ == 'ndarray':
            if not (drivingH.dtype.name == 'complex128'):
                raise ValueError('System Hamiltonian must be passed as a contiguousarray ndarray of dtype complex128')
            else:
                def Hamiltonian(t, drivingH_=drivingH):
                    return drivingH_
                self.H = Hamiltonian
                self.drivingH = drivingH
        else:
            if not (drivingH(0).dtype.name == 'complex128'):
                raise ValueError('System Hamiltonian must be passed as a function returning a contiguousarray ndarray of dtype complex128')
            else:
                self.timedepent_hamiltonian = True
                self.H = drivingH


    def setup_unraveling(self, J_list, TMatrix, c_fields):
        """
        Unraveling setup
        """
        if not type(J_list).__name__ == 'ndarray':
            raise ValueError('The list of jump operators must be passed as a ndarray of dtype complex128')
        else:
            if not (J_list.dtype.name == 'complex128'):
                raise ValueError('The list of jump operators must be passed as a ndarray of dtype complex128')
            
        self.J_list = J_list
        self.num_J = np.shape(self.J_list)[0]

        if TMatrix is not None:
            if check_TMatrix(TMatrix, self.num_J):
                self.T_matrix = TMatrix
        else: 
            self.T_matrix = default_TMatrix(self.num_J)

        self.oT_matrix, self.Tcan_matrix, self.Y_matrix, self.Theta_matrix, self.U_matrix = decomposition_TMatrix(self.T_matrix)

        self.L_list = np.array([sum(Tm*self.J_list) for Tm in self.T_matrix])

        if c_fields is not None:
            if check_cFields(c_fields,self.num_J):
                self.c_fields = np.array([np.eye(self.dimH, dtype=np.complex128)*c_fields[i] for i in np.arange(2*self.num_J)])
        else:
            self.c_fields = default_cFields(self.dimH, self.num_J)

    def setup_jumpy_methods(self):
        self.jumpy_methods = JumpyMethods(self)

    def unitary_analytical(self, integrator='scipy', method='BDF',
                           rrtol=1e-5, aatol=1e-5, last_point=False):
        hamiltonian = self.H
        if self.initial_state_type == 0:
            op_gen_Uni = lambda psi_it, it: schrodinger_operator(hamiltonian, psi_it, it)
            initialState = self.initialStatePsi
        elif self.initial_state_type == 1:
            op_gen_Uni = lambda rho_it, it: vonneumann_operator(hamiltonian, rho_it, it)
            initialState = self.initialStateRho
        if integrator == 'scipy':
            return scipy_integrator(op_gen_Uni, initialState, self.timeList, method=method, rrtol=rrtol, aatol=aatol, last_point=last_point)
        elif integrator == 'runge-kutta':
            return custom_rungekutta_integrator(op_gen_Uni, initialState, self.timeList, last_point=last_point)
        
    def unconditional_analytical(self, integrator='scipy', method='BDF', rrtol=1e-5, aatol=1e-5, last_point=False):
        hamiltonian = self.H
        lindblad_ops = self.J_list
        op_lind = lambda rho_it, it: GKSL_operator(hamiltonian, lindblad_ops, rho_it, it)
        if integrator == 'scipy':
            return scipy_integrator(op_lind, self.initialStateRho, self.timeList, method=method, rrtol=rrtol, aatol=aatol, last_point=last_point)
        elif integrator == 'runge-kutta':
            return custom_rungekutta_integrator(op_lind, self.initialStateRho, self.timeList, last_point=last_point)
        
class JumpyMethods:
    def __init__(self, system):
        self.system = system

    def jump_trajectory(self, seed=0):
        return partial(jumpTrajectory_, self.system.initialStatePsi, 
                       self.system.timeList, 
                       self.system.drivingH, self.system.L_list, self.system.c_fields)(seed)
    
    def jump_traj_ensemble(self, n_trajectories):
        print('Evaluating the jump trajectories ensemble ...')
        all_traj = parallel_run(partial(self.jump_trajectory), np.arange(n_trajectories))
        return all_traj

def check_TMatrix(TMatrix, num_J):
    validity = False

    if not type(TMatrix).__name__ == 'ndarray':
            raise ValueError('Invalid T matrix. The T matrix must be a ndarray of dtype complex128')
    elif not (TMatrix.dtype.name == 'complex128'):
        raise ValueError('Invalid T matrix. The T matrix must be a ndarray of dtype complex128')
    elif np.shape(TMatrix) != (2*num_J, num_J):
        raise ValueError(f"Invalid T matrix. T must have dimesions ({2*num_J},{num_J})")
    
    TdagT = np.round(np.matmul(np.transpose(np.conjugate(TMatrix)),TMatrix), 6)
    diagv_TdagT = np.diag(TdagT)
    diagm_TdagT = np.diag(diagv_TdagT)

    if ((TdagT - diagm_TdagT) != 0).any():
        print('T^dag.T=', repr(TdagT))
        raise ValueError('Invalid T matrix. T must be defined in such a way to ensure that T^dag.T is diagonal')
    elif (np.round(np.real(diagv_TdagT), 10) > 1).any() or (np.round(np.real(diagv_TdagT), 8) < 0).any() or (np.round(np.imag(diagv_TdagT), 8) != 0).any():
        print('T^dag.T=', repr(TdagT))
        raise ValueError('Invalid T matrix. T must be defined in such a way to ensure that 0 <= T^dag.T <= 1, representing the detection efficiency.')
    else:
        validity = True

    return validity

def check_cFields(c_fields, num_J):
    validity = False

    if not type(c_fields).__name__ == 'ndarray':
            raise ValueError('Invalid c-fields. The c-fields must be a ndarray of dtype complex128')
    elif not (c_fields.dtype.name == 'complex128'):
        raise ValueError('Invalid c-fields. The c-fields must be a ndarray of dtype complex128')
    elif np.shape(c_fields)[0] != 2*num_J:
        raise ValueError(f"Invalid c-fields. The c-fields must have dimesions {2*num_J}")
    
    if (np.round(np.imag(c_fields), 8) != 0).any():
        raise ValueError('Invalid c-fields. the c-fields must be real.')
    else:
        validity = True
    return validity

def default_TMatrix(N):
    sm = np.ascontiguousarray(np.zeros((N,2*N),dtype=np.complex128))
    for i in np.arange(N):
        a = np.zeros(2*i)
        b = np.array([1, 0])
        c = np.zeros(2*(N-1-i))
        sm[i] = np.block([a,b,c])
    return np.transpose(sm)

def default_cFields(dimH,N):
    return np.zeros((2*N,dimH,dimH), dtype=np.complex128)
    
def decomposition_TMatrix(T):
    dimN = T.shape[1]
    TT = np.block([T.real, T.imag])
    oT, p = polar(TT)
    Tcan = p[:,:dimN] + 1j*p[:,dimN:]
    Ym = np.matmul(np.transpose(np.conjugate(Tcan)),np.conjugate(Tcan))
    Thetam = np.matmul(np.transpose(np.conjugate(Tcan)),Tcan)
    Um = np.matmul(np.transpose(TT),TT)
    return oT, Tcan, Ym, Thetam, Um


