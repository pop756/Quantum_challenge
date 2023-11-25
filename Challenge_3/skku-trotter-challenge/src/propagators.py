import numpy as np

from qiskit import QuantumCircuit


class UXZGate(QuantumCircuit):
    r"""
    Construct a ``QuantumCircuit`` implementing the block operator
    $U_{XZ}(\gamma, \delta)$ describing time evolution under

    .. math::

        H_X + X_Z = \frac{\gamma}{2} \sigma_0^X \sigma_1^X + \frac{\delta}{2} \sigma_0^Z \sigma_1^Z.

    The parameters $\gamma$ and $\delta$ can be of type ``float`` or
    ``qiskit.circuit.Parameter``.

    EXAMPLES::


        >>> from qiskit.circuit import Parameter
        >>> gamma, delta = Parameter("gamma"), Parameter("delta")
        >>> UXZGate(gamma, delta).draw()
                  ┌───────────┐     
        q_0: ──■──┤ Rx(gamma) ├──■──
             ┌─┴─┐├───────────┤┌─┴─┐
        q_1: ┤ X ├┤ Rz(delta) ├┤ X ├
             └───┘└───────────┘└───┘
    """
    def __init__(self, gamma, delta):
        super().__init__(2, name="Uxz")
        self.cnot(0,1)
        self.rx(gamma,0)
        self.rz(delta,1)
        self.cx(0,1)  


class UXYZGate(QuantumCircuit):
    r"""
    Construct a ``QuantumCircuit`` implementing the block operator
    $U_{XYZ}(\theta)$ describing evolution according to the the Heisenberg
    $XYZ$-Hamiltonian

    .. math::

        H_{XYZ} = \frac{1}{2} \sum_{\alpha \in \{X, Y, Z\}} \theta_\alpha \, \sigma_0^\alpha \sigma_1^\alpha.

    Here ``thetax``, ``thetay``, and ``thetaz`` can be of type ``float`` or
    ``qiskit.circuit.Parameter``.

    EXAMPLES:

        >>> from qiskit.circuit import ParameterVector
        >>> theta = ParameterVector("t", 3)
        >>> UXYZGate(*theta).draw()
                  ┌──────────┐┌───┐           ┌───┐      ┌───┐     ┌──────────┐
        q_0: ──■──┤ Rx(t[0]) ├┤ H ├──■────────┤ S ├──────┤ H ├──■──┤ Rx(-π/2) ├
             ┌─┴─┐├──────────┤└───┘┌─┴─┐┌─────┴───┴─────┐└───┘┌─┴─┐├─────────┬┘
        q_1: ┤ X ├┤ Rz(t[2]) ├─────┤ X ├┤ Rz(-1.0*t[1]) ├─────┤ X ├┤ Rx(π/2) ├─
             └───┘└──────────┘     └───┘└───────────────┘     └───┘└─────────┘ 
    """
    def __init__(self, thetax, thetay, thetaz):
        super().__init__(2, name="Uxyz")
        self.cnot(0,1)
        self.rx(thetax,0)
        self.rz(thetaz,1)
        self.h(0)
        self.cnot(0,1)
        self.s(0)
        self.rz(-1.*thetay)
        self.h(0)
        self.cnot(0,1)
        self.rx(0,-np.pi/2)
        self.rx(1,np.pi/2)

