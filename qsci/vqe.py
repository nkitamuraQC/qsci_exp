import pennylane as qml
import numpy as np
from pennylane import numpy as qnp

class UCCSD_Lattice:
    def __init__(self, int1e, int2e, norb, nelec):
        """
        UCCSD class
        
        Args:
            int1e (np.ndarray): one-electron integrals
            int2e (np.ndarray): two-electron integrals
            norb (int): number of orbitals
            nelec (int): number of electrons
        """
        self.int1e = int1e
        self.int2e = int2e
        self.norb = norb
        self.nelec = nelec

        self.qubits = norb * 2

        self.hf_state = qml.qchem.hf_state(self.nelec, self.qubits)
        self.h = None

        if int1e is not None and int2e is not None:
            self.h1e = qml.qchem.one_particle(int1e)
            self.h2e = qml.qchem.two_particle(int2e)
            self.h = qml.qchem.observable([self.h1e, self.h2e], mapping="jordan_wigner")

        singles, doubles = qml.qchem.excitations(self.nelec, self.qubits)
        print(singles)

        self.singles = singles
        self.doubles = doubles

        # Map excitations to the wires the UCCSD circuit will act on
        self.s_wires, self.d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        self.shots = 1000
        # self.shots = 50000

        # Define the device
        self.dev = qml.device("default.qubit", wires=self.qubits)
        self.dev_shot = qml.device("default.qubit", wires=self.qubits, shots=self.shots)

        # Define the initial values of the circuit parameters
        self.params = qnp.random.rand(len(singles) + len(doubles))

    def optimize(self):
        """
        Optimize UCCSD parameters
        """
        @qml.qnode(self.dev)
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.expval(self.h)

        @qml.qnode(self.dev)
        def get_state(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.state()

        # Define the optimizer
        optimizer = qml.GradientDescentOptimizer(stepsize=0.5)

        # Optimize the circuit parameters and compute the energy
        for n in range(11):
            self.params, energy = optimizer.step_and_cost(
                circuit,
                self.params,
                wires=range(self.qubits),
                s_wires=self.s_wires,
                d_wires=self.d_wires,
                hf_state=self.hf_state,
            )
            if n % 2 == 0:
                print("step = {:},  E = {:.8f} Ha".format(n, energy))
        return

    def sample(self):
        """
        Sample from the supplied observable, with the number of shots determined 
        from the dev.shots attribute of the corresponding device, returning raw samples. 
        If no observable is provided then basis state samples are returned directly from the device.

        Returns:
            qml.measurements.SampleMP: results of sampling
        """
        @qml.qnode(self.dev_shot)
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.sample()

        smp = circuit(
            self.params,
            wires=range(self.qubits),
            s_wires=self.s_wires,
            d_wires=self.d_wires,
            hf_state=self.hf_state,
        )
        return smp
