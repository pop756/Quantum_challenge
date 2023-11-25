import numpy as np

from qiskit.quantum_info import SparsePauliOp


class StaggeredMagnetization:
    r"""
    Calculator for the time-dependent staggered magnetization of a spin
    chain evolving under the action of a Heisenberg Hamiltonian.

    In particular, this class implements a ``__call__`` dunder method that
    evaluates the expectation value defining $M_{\mathrm{stag}}(t)$ with
    respect to the quantum state described by the ``measurements``
    dictionary of ``(state, counts)`` key-value pairs.

    EXAMPLES::

        >>> from qiskit_aer import Aer
        >>> from src.staggered_magnetization import StaggeredMagnetization
        >>> from src.xyz_evolution import XYZEvolutionCircuit
        >>> num_qubits, t = 3, 1
        >>> qc = XYZEvolutionCircuit(num_qubits, [1, 1, 1], final_time=t, trotter_num=10)
        >>> backend = Aer.get_backend("aer_simulator")
        >>> qc = qc.decompose(["Uxz", "Uxyz"])
        >>> qc.measure_all()
        >>> psi = backend.run(qc).result().get_counts()
        >>> m_stag = StaggeredMagnetization(num_qubits)
        >>> m_stag(psi)
        0.3333333333333333
    """
    def __init__(self, num_qubits):
        terms = [("Z", [j], (-1)**j) for j in range(num_qubits)]
        self.hamiltonian = 1 / num_qubits * SparsePauliOp.from_sparse_list(terms, num_qubits)

    def __call__(self, measurements):
        r"""
        Evaluate the time-dependent staggered magnetization $m_s(t)$ defined by
        :ref:`Equation (1) <stag_mag>` with respect to the quantum state
        $\vert \psi(t) \rangle$ described by the dictionary ``measurements``
        of ``(state, count)`` key-value pairs.
        """
        n = self.hamiltonian.num_qubits
        dtype = np.dtype([("states", int, (n,)), ("counts", "f")])
        res = np.fromiter(map(lambda kv: (list(kv[0]), kv[1]), measurements.items()), dtype)
        shots = res["counts"].sum()
        return np.dot(self.energies(res["states"]), res["counts"]) / shots

    def energies(self, states):
        """
        Quickly obtain eigenvalues of an Ising Hamiltonian corresponding to the
        given ``states``.

        EXAMPLES::

            >>> import numpy as np
            >>> from src.staggered_magnetization import StaggeredMagnetization
            >>> m_stag = StaggeredMagnetization(5)
            >>> states = np.array([[0, 1, 1, 0, 1], [1, 1, 0, 0, 0]])
            >>> m_stag.energies(states)
            array([-0.2,  0.2])
        """
        paulis = np.array([list(str(ops)) for ops in self.hamiltonian.paulis]) != "I"
        coeffs = self.hamiltonian.coeffs.real
        energies = [0]*len(states)
        
        #####################
        ### Fill this in! ###
        #####################

        return energies
