import numpy as np

from itertools import product
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm
from src.circuit_compressor import CircuitCompressor
from src.propagators import UXZGate, UXYZGate
from src.staggered_magnetization import StaggeredMagnetization
from src.xyz_evolution import XYZEvolutionCircuit


class XYZEvolutionTestSuite:
    """
    A class to test your implementation of the various phases of this challenge.
    """
    def __init__(self):
        pass 

    def run_test_suite(self):
        self.test_uxz()
        self.test_uxyz()
        self.test_xyz_evolution()
        self.test_xyzm_evolution()
        self.test_magnetization_hamiltonian()
        self.test_ybe_update()
        self.test_mirror_update()
        self.test_circuit_compression()

    def test_magnetization_hamiltonian(self, num_qubits=7):
        for n in range(1, num_qubits+1):
            ms = StaggeredMagnetization(n)

            # Compute all states
            a = np.arange(2**n, dtype=int)[:, np.newaxis]
            b = np.arange(n, dtype=int)[np.newaxis, ::-1]
            states = np.array(2**b & a > 0, dtype=int)

            # Check that eigenvalues match
            hamiltonian = 1/n * SparsePauliOp.from_sparse_list([("Z", [j], (-1)**j) for j in range(n)], n)
            assert np.allclose(ms.energies(states), np.diag(hamiltonian.to_matrix())), "Test failed..."
        print("Magnetization test passed!")

    def test_uxz(self, num_points=10):
        r"""
        Test your implementation of the $B_{XZ}$ block.
        """
        for gamma, delta in product(np.linspace(-2*np.pi, 2*np.pi, num_points), repeat=2):
            self._test_time_propagator_circuit(UXZGate(gamma, delta), [gamma, 0, delta])
        print("Uxz test passed!")

    def test_uxyz(self, num_points=10):
        r"""
        Test your implementation of the $B_{XYZ}$ block.
        """
        for theta in product(np.linspace(-2*np.pi, 2*np.pi, num_points), repeat=3):
            self._test_time_propagator_circuit(UXYZGate(*theta), theta)
        print("Uxyz test passed!")

    def test_xyz_evolution(self, num_qubits=5, num_points=3):
        """
        Test your implementation of the $XYZ$ time propagator.
        """
        for n in range(2, num_qubits+1):
            for J in product(np.linspace(0.1, 1, num_points), repeat=3):
                for t in np.linspace(0, 0.1, num_points):
                    xyz_evolution_qc = XYZEvolutionCircuit(n, np.array(J), final_time=t, trotter_num=15)
                    self._test_time_propagator_circuit(xyz_evolution_qc, J, t=t, tol=1e-2)
        print("XYZ evolution test passed!")

    def test_xyzm_evolution(self, num_qubits=5, num_points=3):
        """
        Test your implementation of the $XYZ$ time propagator under a transverse
        magnetic field.
        """
        for n in range(num_qubits, num_qubits+1):
            for J in product(np.linspace(0.1, 1, num_points), repeat=3):
                for t in np.linspace(0, 0.1, num_points):
                    for m in np.linspace(0, 2, num_points):
                        xyzm_evolution_qc = XYZEvolutionCircuit(n, np.array(J), m, t, 15)
                        self._test_time_propagator_circuit(xyzm_evolution_qc, J, m, t, tol=1e-2)
        print("XYZ evolution with transverse magnetic field test passed!")

    def _test_time_propagator_circuit(self, qc, J, h=0, t=1, tol=1e-5):
        r"""
        Numerically verify that the ``QuantumCircuit`` ``qc`` indeed implements
        the time evolution operator corresponding to the given ``hamiltonian``.

        Concretely, this test if the unitary matrix implemented by ``qc`` is
        close to the unitary matrix $\exp(-i/2 H)$ in Frobenius norm, with $H$
        denoting ``hamiltonian.to_matrix()``.
        """
        n = qc.num_qubits
        site_ham = lambda j: SparsePauliOp.from_sparse_list([(ax * 2, [j, j+1], Jax) for ax, Jax in zip("XYZ", J)], n)
        hamiltonian = SparsePauliOp.sum([site_ham(j) for j in range(n-1)])
        hamiltonian += SparsePauliOp.from_sparse_list([("Z", [j], h) for j in range(n)], n)

        time_prop = expm(-1j/2 * t * hamiltonian.to_matrix())
        assert np.allclose(Operator(qc).data, time_prop, atol=tol), "Test failed..."

    def test_qc_param_update(self, num_qubits, num_points=3):
        propagator = "XZ"

        # Construct LHS and RHS circuits
        lhs = XYZEvolutionCircuit(num_qubits)
        lhs._construct_evolution_qc(propagator, num_layers=num_qubits, bound=False)
        rhs = XYZEvolutionCircuit(num_qubits)
        rhs._construct_evolution_qc(propagator, num_layers=num_qubits, odd=True, bound=False)
        num_blocks = (num_qubits * (num_qubits - 1)) // 2

        # Choose the appropriate updater
        lhs_compressor = CircuitCompressor(lhs)
        updater = lhs_compressor.get_ybe_update if num_qubits == 3 else lhs_compressor.get_mirror_update
        rhs_compressor = CircuitCompressor(rhs)
        rhs_updater = rhs_compressor.get_ybe_update if num_qubits == 3 else rhs_compressor.get_mirror_update

        for _ in range(num_points):
            # Generate random block parameters
            lhs_params = np.random.rand(num_blocks * 2)

            # Compute parameters for equivalent circuit
            rhs_params = updater(lhs_params)
            
            # Verify we obtain an equivalent circuit
            assert Operator(lhs.assign_parameters(lhs_params)) == Operator(rhs.assign_parameters(rhs_params)), "Test failed..."

            # Test right-to-left update
            new_lhs_params = rhs_updater(rhs_params, l2r=False)
            assert Operator(lhs.assign_parameters(new_lhs_params)) == Operator(rhs.assign_parameters(rhs_params)), "Test failed..."

    def test_ybe_update(self, num_points=3):
        """
        Test your implementation of the YBE parameter update step.
        """
        self.test_qc_param_update(3, num_points)
        print("YBE update test passed!")

    def test_mirror_update(self, num_points=3):
        """
        Test your implementation of the mirror update step, which is a sequence
        of YBE updates.
        """
        self.test_qc_param_update(4, num_points)
        print("Mirror update test passed!")

    def test_circuit_compression(self, trotter_num=5):
        """
        Test your implementation of the full compression scheme by checking that
        the unitary operator implemented by your compressed circuit indeed
        matches the expected operation.
        """
        J = np.array([0.8, 0, -0.2])
        t = 0.05
        for num_qubits in range(3, 5):
            qc = XYZEvolutionCircuit(num_qubits, J, final_time=t, trotter_num=trotter_num)
            qc = CircuitCompressor(qc).compress_circuit()
            self._test_time_propagator_circuit(qc, J, t=t, tol=1e-2)
        print("Circuit compression test passed!")
