import pennylane as qml
import numpy as np
from torch.autograd import Variable
import torch, copy
from pyscf import mcscf
from pennylane.templates import QuantumPhaseEstimation


def cgm(multA, b, x_init):
    x = x_init

    r0 = b - multA(x)
    p = r0
    for i in range(10000):
        a = np.dot(r0.T.conj(), r0) / np.dot(p.T.conj(), multA(p))
        x = x + p*a
        #x /= np.linalg.norm(x)
        r1 = r0 - multA(p) * a
        #print(f"residual[{i}] =", np.linalg.norm(r1))
        #print("norm =", np.linalg.norm(x))
        if np.linalg.norm(r1) < 1.0e-10:
            return x
        b = np.dot(r1.T.conj(), r1) / np.dot(r0.T.conj(), r0)
        p = r1 + b * p
        r0 = r1
    print("Not Converged")
    return x

class UCCSD_Lattice:
    def __init__(self, int1e, int2e, norb, nelec):
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
        #self.shots = 50000

        # Define the device
        self.dev = qml.device("default.qubit", wires=self.qubits)
        self.dev_shot = qml.device("default.qubit", wires=self.qubits, shots=self.shots)

        # Define the initial values of the circuit parameters
        self.params = np.random.rand(len(singles) + len(doubles))

        self.T1 = None
        self.T2 = None

    def apply(self):
        @qml.qnode(self.dev)
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.state()
        state = circuit(self.params, wires=range(self.qubits), s_wires=self.s_wires, \
                        d_wires=self.d_wires, hf_state=self.hf_state)
        
        return state
    
    def input_graph(self, adjmat):
        @qml.qnode(self.dev)
        def circuit(params, wires, s_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, None, hf_state)
            return qml.state()
        
        adjmat = np.reshape(adjmat, (1, -1))
        
        state = circuit(adjmat, wires=range(self.qubits), s_wires=self.s_wires, \
                        hf_state=self.hf_state)
        
        return state



    def optimize(self):
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
            self.params, energy = optimizer.step_and_cost(circuit, self.params, wires=range(self.qubits), s_wires=self.s_wires, d_wires=self.d_wires, hf_state=self.hf_state)
            if n % 2 == 0:
                print("step = {:},  E = {:.8f} Ha".format(n, energy))
        state = get_state(self.params, range(self.qubits), self.s_wires, self.d_wires, self.hf_state)
        return energy, state
    
    def get_grad(self):
        @qml.qnode(self.dev, interface='torch')
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.expval(self.h)
 
        params = Variable(torch.tensor(self.params), requires_grad=True)
        result = circuit(params, wires=range(self.qubits), s_wires=self.s_wires, d_wires=self.d_wires, hf_state=self.hf_state)
        result.backward()
        grad = params.grad
        return grad
    
    def get_second_grad(self):
        @qml.qnode(self.dev, interface='torch')
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.expval(self.h)
 
        params = Variable(torch.tensor(self.params), requires_grad=True)
        result = circuit(params, wires=range(self.qubits), s_wires=self.s_wires, d_wires=self.d_wires, hf_state=self.hf_state)
        g = torch.autograd.grad(result, params)
        g[0].backward()
        grad = params.grad
        return grad
    
    def coupling(self, int1e_grad=None, int2e_grad=None):
        tt = self.get_second_grad()
        if int1e_grad is not None and int2e_grad is not None:
            self.h1e_grad = qml.qchem.one_particle(int1e)
            self.h2e_grad = qml.qchem.two_particle(int2e)
            self.h_grad = qml.qchem.observable([self.h1e, self.h2e], mapping="jordan_wigner")
        h = copy.deepcopy(self.h)
        self.h = copy.deepcopy(self.h_grad)
        self.optimize()
        t = self.get_grad()
        couple = np.linalg.solve(tt, -t)
        self.h = copy.deepcopy(h)
        return couple



    ### 拡張すれば任意の粒子数の密度行列計算可能 (i, j, k, l)
    def get_dm1e(self, i, j):
        coeff_1, op_1 = qml.qchem.jordan_wigner([i, j])
        coeff_2, op_2 = qml.qchem.jordan_wigner([j, i])
        op = qml.Hamiltonian(coeff_1+coeff_2, op_1+op_2)
        #op2 = qml.Hamiltonian(coeff_2, op_2)
        #op = op1 + op2
        @qml.qnode(self.dev)
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.expval(op)
        
        dm = circuit(self.params, wires=range(self.qubits), s_wires=self.s_wires, d_wires=self.d_wires, hf_state=self.hf_state) / 2
        return dm
    
    def sample(self):
        @qml.qnode(self.dev_shot)
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.sample()
        
        smp = circuit(self.params, wires=range(self.qubits), s_wires=self.s_wires, d_wires=self.d_wires, hf_state=self.hf_state)
        return smp
    
    def get_Trdm1e(self, i, j, state0, state1):
        coeff_1, op_1 = qml.qchem.jordan_wigner([i, j])
        coeff_2, op_2 = qml.qchem.jordan_wigner([j, i])
        op = qml.Hamiltonian(coeff_1+coeff_2, op_1+op_2)
        @qml.qnode(self.dev)
        def circuit(state0):
            qml.QubitStateVector(state0, wires=range(self.qubits))
            qml.apply(op)
            return qml.state()
        
        vec = circuit(state0)
        dm = np.dot(state1, vec)
        return dm
    
    def apply_cre(self, state, i):
        @qml.qnode(self.dev)
        def circuit_X(state, i):
            qml.QubitStateVector(state, wires=range(self.qubits))
            qml.apply(qml.PauliX(wires=i))
            for k in range(i):
                qml.apply(qml.PauliZ(wires=k))
            return qml.state()
        
        @qml.qnode(self.dev)
        def circuit_Y(state, i):
            qml.QubitStateVector(state, wires=range(self.qubits))
            qml.apply(qml.PauliY(wires=i))
            for k in range(i):
                qml.apply(qml.PauliZ(wires=k))
            return qml.state()
        
        state_X = circuit_X(state, i)
        state_Y = circuit_Y(state, i) * 1.0j
        return (state_X + state_Y) * 0.5
    
    def apply_des(self, state, i):
        @qml.qnode(self.dev)
        def circuit_X(state, i):
            qml.QubitStateVector(state, wires=range(self.qubits))
            qml.apply(qml.PauliX(wires=i))
            for k in range(i):
                qml.apply(qml.PauliZ(wires=k))
            return qml.state()
        
        @qml.qnode(self.dev)
        def circuit_Y(state, i):
            qml.QubitStateVector(state, wires=range(self.qubits))
            qml.apply(qml.PauliY(wires=i))
            for k in range(i):
                qml.apply(qml.PauliZ(wires=k))
            return qml.state()
        
        state_X = circuit_X(state, i)
        state_Y = - circuit_Y(state, i) * 1.0j
        return (state_X + state_Y) * 0.5
    
    def sigma(self, state):
        @qml.qnode(self.dev)
        def circuit_H(state):
            qml.QubitStateVector(state, wires=range(self.qubits))
            qml.apply(self.h)
            return qml.state()
        
        state = circuit_H(state)
        return state
    
    def get_green(self, state, E, omega, i, j, op_type="cre"):
        if op_type == "cre":
            b = self.apply_cre(state, i)
            c = self.apply_cre(state, j)
        if op_type == "des":
            b = self.apply_des(state, i)
            c = self.apply_des(state, j)
        self.E = E
        self.omega = omega
        self.eta = 0.1
        x_init = np.zeros_like(b)
        X1 = cgm(self.multA, b, x_init)
        Xvec = (self.sigma(X1) - (self.X) * X1) / self.eta
        Yvec = X1
        G_re = np.dot(c, Xvec)
        G_im = np.dot(c, Yvec)

        return G_re, G_im

    def multA(self, x):
        X = self.E + self.omega 
        self.X = X
        vec = X**2 * x - 2*X * self.sigma(x, ca=self.ca) + self.sigma(self.sigma(x, ca=self.ca), ca=self.ca) + (self.eta ** 2) * x
        return vec
    
    def TPQ(self, l=1.0, max_cycle=100):
        state = self.apply()
        for i in range(max_cycle):
            state = l * state - self.sigma(state)
            state /= np.linalg.norm(state)
        E = self.Energy(state)
        return state
    
    def Energy(self, state):
        @qml.qnode(self.dev)
        def circuit_H(state):
            qml.QubitStateVector(state, wires=range(self.qubits))
            return qml.expval(self.h)
        return circuit_H(state)



    
if __name__ == "__main__":
    """
    norb = 2
    nelec = 2
    int1e = np.zeros((norb, norb))
    int2e = np.zeros((norb, norb, norb, norb))
    uccsd = UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.optimize()
    smp = uccsd.sample()
    print(smp)
    state0 = np.zeros((2**(norb*2)))
    state1 = np.zeros((2**(norb*2)))
    state0[0] = 1
    state1[0] = 1
    uccsd.get_Trdm1e(0, 0, state0, state1)
    """
    norb = 2
    nelec = 2
    int1e = np.zeros((norb, norb))
    int2e = np.zeros((norb, norb, norb, norb))
    uccsd = UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.qpe_circuit()
