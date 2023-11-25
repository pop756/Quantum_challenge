import numpy as np
import random as rnd

from copy import deepcopy
from qiskit.quantum_info import Operator
from scipy.optimize import minimize
from src.xyz_evolution import XYZEvolutionCircuit


class CircuitCompressor:
    """
    A class that implements the YBE-powered QTD circuit compression scheme.
    """
    def __init__(self, xyz_evolution_qc):
        self.deep_qc = xyz_evolution_qc

    def get_ybe_update(self, params, l2r=True):
        r"""
        Update $3$-qubit time propagator circuit parameters according to the
        Yang-Baxter Equation (YBE).


        INPUT:

            - ``params`` -- a (3, 2) array describing the parameters for each
              of the three blocks
            - ``l2r`` -- a boolean indicating whether we are applying the symmetry
              left-to-right or vice versa

        OUTPUT:

            A (3, 2) NumPy array describing the parameters of an equivalent circuit. 
        """
        # Get equivalent parametrized circuit
        equiv_qc = XYZEvolutionCircuit(3)
        
        #####################
        ### Fill this in! ###
        #####################

        # Construct target unitary operator
        target_qc = XYZEvolutionCircuit(3)
        
        #####################
        ### Fill this in! ###
        #####################

        target = Operator(target_qc).data

        # Get parameters for the equivalent time propagator
        new_params = get_optimized_params(equiv_qc, target)
        return new_params

    def get_mirror_update(self, params, l2r=True):
        r"""
        Update the $4$-qubit time propagator circuit parameters following the
        so-called mirror step.

        INPUT:

            - ``params`` -- a (6, 2) NumPy array describing the parameters for
              each of the six blocks
            - ``l2r`` -- a boolean indicating whether we are applying the symmetry
              left-to-right or vice versa
        
        OUTPUT:

            A (6, 2) NumPy array describing the parameters of the mirrored circuit.
        """
        # Number of parameters per block
        bsz = 2
        
        #####################
        ### Fill this in! ###
        #####################

        return params

    def compress_circuit(self):
        """
        Compress this time evolution circuit using the YBE.

        OUTPUT:

            Returns a compressed :class:`.XYZEvolutionCircuit` object
            with $N (N - 1)/2$ blocks that is equivalent to the uncompressed
            Trotterization circuit ``self.deep_qc``.
        """
        # Return if deep_qc is empty
        qc = self.deep_qc
        if not list(qc):
            return qc

        # Extract parameters
        bsz = 2
        evol_params = qc.time_delta * qc.coupling_const[np.nonzero(qc.coupling_const)]
        N = qc.num_qubits
        w = np.zeros(N*(N-1))

        #####################
        ### Fill this in! ###
        #####################

        # Construct compressed circuit
        compressed = XYZEvolutionCircuit(N)
        compressed._construct_evolution_qc(qc.propagator, num_layers=N, bound=False)
        return compressed.assign_parameters(w)

#########################
### Parameter fitting ###
#########################

# Minimize quadratic target loss... Attempt max_shot times using random initial point
def get_optimized_params(param_circ, target_unitary, res_tol=1e-7, max_shots=10, verbose=False):
    # Define quadratic loss for fitting
    loss = lambda p: np.linalg.norm(Operator(param_circ.assign_parameters(p)).data - target_unitary)
    n_params = param_circ.num_parameters
    min_loss = 100
    w_opt = 0
    for k in range(max_shots):
        w0 = [2*np.pi*rnd.random()+np.pi for j in range(n_params)]
        opt = minimize(
                loss, w0, method='L-BFGS-B', bounds=[(0, 4*np.pi)]*n_params,
                jac='3-point', options={'ftol': 1e-10}
            )
        if opt.fun < min_loss:
            min_loss = opt.fun
            w_opt = opt.x
        if min_loss < res_tol:
            return w_opt
    if verbose:
        print("WARNING: Good fitting parameters were not found! Residual loss is {}".format(min_loss))
    return w_opt
