import pennylane as qml
from pennylane import ApproxTimeEvolution
from qsci import calc_weights
import numpy as np
from pennylane import numpy as qnp


def separate_constant_term(hamiltonian):
    # 定数項を初期化
    constant = 0
    new_coeffs = []
    new_ops = []

    # ハミルトニアンを走査
    for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops):
        if isinstance(op, qml.Identity):  # 定数項をチェック
            constant += coeff
        else:
            new_coeffs.append(coeff)
            new_ops.append(op)

    # 新しいハミルトニアンを作成
    new_hamiltonian = qml.Hamiltonian(new_coeffs, new_ops)

    return constant, new_hamiltonian


class InvHam:
    def __init__(self, ham, nqubits):
        self.ham = ham
        self.const, self.hact = separate_constant_term(self.ham)
        self.nqubits = nqubits
        self.dev = qml.device("default.qubit", wires=range(self.nqubits))
        self.weights = calc_weights.CalcWeights()
        self.ny = self.weights.ny
        self.nz = self.weights.nz
        self.delta_y = self.weights.delta_y
        self.delta_z = self.weights.delta_z
        self.max_y = self.weights.max_y
        self.nz = self.weights.nz
        self.max_z = self.weights.max_z
        self.min_z = self.weights.min_z
        self.k_param = 10
        self.expH = self.exp()
        self.prepare_circuits()

    def exp(self):
        @qml.qnode(self.dev)
        def expH(statevec):
            qml.QubitStateVector(statevec, wires=range(self.nqubits))
            return qml.expval(self.ham)

        return expH

    def time_evol(self, t, s, n=20):
        @qml.qnode(self.dev)
        def circuit(t, s):
            if s is not None:
                qml.QubitStateVector(s, wires=range(self.nqubits))
            ApproxTimeEvolution(self.ham, t, n)
            return qml.state()

        return circuit(t, s)

    def prepare_circuits(self):
        self.ws = []
        self.ts = []
        for jy in range(self.ny):
            for jz in range(self.nz):
                w = self.weights.get_weights_k(jy, jz, self.k_param)
                self.ws.append(w)
                t = jy * self.delta_y * (jz * self.delta_z + self.min_z)
                self.ts.append(t)
        return

    def weighted_inv(self, s, k=30):
        # print(s)
        E = self.expH(s)
        print(-1, E)
        for niter in range(k):
            s_save = qnp.zeros((self.nqubits**2), dtype=complex)
            for i in range(len(self.ws)):
                t = self.ts[i]
                s_save += np.array(self.time_evol(t, s)) * self.ws[i]
            s = qnp.array(s_save)
            s /= np.linalg.norm(s)
            # print(s)
            E = self.expH(s)
            print(niter, E)
        return E
