import pennylane as qml
import numpy as np
from qsci.vqe import UCCSD_Lattice
import copy, math
from pyscf.fci import cistring
from pyscf.fci.selected_ci import kernel_fixed_space, _as_SCIvector, SelectedCI


def list2int(val):
    """
    Calculates a decimal number from a binary number given by python list.

    Args:
        val (list[int]): a binary number given by python list
            
    Returns:
        int: a decimal number
    """
    ret = 0
    for i, v in enumerate(val):
        ret += (2**i) * v
    return ret


def state2occ(state, norb):
    """
    Calculates occupation number vector from qubit state vector.

    Args:
        state (list[int]): a state
            
    Returns:
        list[int]: an occupation number vector
    """
    index = np.argmax(state)
    occ_str = bin(index)[2:].zfill(norb * 2)[::-1]
    occ = []
    for o in range(len(occ_str)):
        if occ_str[o] == "1":
            occ.append(o)
    return occ


def qubit2rhf(occ, norb, nelec):
    """
    Calculates a RHF CI vector.

    Args:
        occ (list[int]): an occupation number vector
        norb (int): number of orbitals
        nelec (int): number of electrons
            
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: RHF CI vector, occupation number for alpha electrons and beta electrons
    """
    # _occslst2strs(occslst)の使い方を調べる
    occ_alpha = np.array([occ[o] // 2 for o in range(len(occ)) if occ[o] % 2 == 0])
    occ_beta = np.array([occ[o] // 2 for o in range(len(occ)) if occ[o] % 2 == 1])
    ndet = math.comb(norb, nelec // 2)
    c = np.zeros((ndet, ndet))
    occlst = cistring.gen_occslst(range(norb), nelec // 2)
    for i, occlst_ in enumerate(occlst):
        occlst_arr = np.array(occlst_)
        if np.linalg.norm(occ_alpha - occlst_arr[0]) < 1e-6:
            break

    for j, occlst_ in enumerate(occlst):
        occlst_arr = np.array(occlst_)
        if np.linalg.norm(occ_beta - occlst_arr[0]) < 1e-6:
            break

    c[i, j] = 1
    return c, occ_alpha, occ_beta


class Sampler:
    def __init__(self, model):
        """
        Sampler class
        
        Args:
            model (vqe.UCCSD): UCCSD class
        """
        self.model = model

    def sampling(self):
        """
        Sample from the supplied observable, with the number of shots determined 
        from the dev.shots attribute of the corresponding device, returning raw samples. 
        If no observable is provided then basis state samples are returned directly from the device.

        Returns:
            qml.measurements.SampleMP: results of sampling
        """
        sample = self.model.sample()
        return sample

    def calc_freq(self, sample):
        """
        Get frequencies
        
        Args:
            np.ndarray: frequencies
        """
        norb = self.model.int1e.shape[0]
        freq = np.zeros((2 ** (norb * 2)))
        for i in range(sample.shape[0]):
            index = list2int(sample[i, :])
            freq[index] += 1
        return freq


class QSCI:
    def __init__(self, sampler, int1e=None, int2e=None, norb=None, nelec=None):
        """
        QSCI class
        
        Args:
            sampler (qsci.qsci.sampler): the sampler class
            int1e (np.ndarray): one-electron integrals
            int2e (np.ndarray): two-electron integrals
            norb (int): number of orbitals
            nelec (int): number of electrons
        """
        self.int1e = sampler.model.int1e
        self.int2e = sampler.model.int2e
        self.norb = sampler.model.norb
        self.nelec = sampler.model.nelec
        self.sampler = sampler
        self.nchoose = 4

        self.dev = qml.device("default.qubit", wires=self.norb * 2)

    def choose(self):
        """
        Chooses important electronic configurations.
                
        Returns:
            np.ndarray: important electronic configurations
        """
        sample = self.sampler.sampling()
        freq = self.sampler.calc_freq(sample)
        freq_large = np.argsort(freq)[::-1][: self.nchoose]
        return freq_large

    def arr2qubit(self):
        """
        Builds a qubit Hamiltonian.
                
        Returns:
            qml.qchem.observable: a qubit Hamiltonian
        """
        self.h1e = qml.qchem.one_particle(self.int1e)
        self.h2e = qml.qchem.two_particle(self.int2e)
        h_ob = qml.qchem.observable([self.h1e, self.h2e], mapping="jordan_wigner")
        return h_ob

    def freqindex2qubit(self, freq_index):
        """
        Builds a qubit vector from important electronic configurations.
                
        Returns:
            np.ndarray: a qubit vector
        """
        ret = []
        for i in range(len(freq_index)):
            state = np.zeros((2 ** (self.norb * 2)))
            state[freq_index[i]] = 1
            ret.append(state)
        ret = np.array(ret)
        return ret


    def diagonalize_sci(self, nroots=1):
        """
        Diagonalize the Hamiltonian matrix by selected CI algorithm

        Args:
            nroots (int): index of roots
                
        Returns:
            tuple[float, np.ndarray]: electronic energy, a CI vector
        """
        freq_large = self.choose()
        states = self.freqindex2qubit(freq_large)
        norb = self.norb
        nelec = self.nelec
        occ_a = []
        occ_b = []
        for i in range(len(states)):
            occ_i = state2occ(states[i], norb)
            if len(occ_i) != self.nelec:
                continue
            _, occ_alpha, occ_beta = qubit2rhf(occ_i, norb, nelec)
            if len(occ_alpha) != self.nelec // 2:
                continue
            if len(occ_beta) != self.nelec // 2:
                continue
            occ_a.append(occ_alpha.tolist())
            occ_b.append(occ_beta.tolist())
        strs_a = cistring._occslst2strs(np.asarray(occ_a, dtype=np.int32).view(cistring.OIndexList))
        strs_b = cistring._occslst2strs(np.asarray(occ_b, dtype=np.int32).view(cistring.OIndexList))
        ci_strs = (list(set(strs_a)), list(set(strs_b)))
        occ_0 = state2occ(states[i], norb)
        ci0, _, _ = qubit2rhf(occ_0, norb, nelec)
        c1 = _as_SCIvector(ci0, ci_strs)
        myci = SelectedCI()

        e, c = kernel_fixed_space(
            myci, self.int1e, self.int2e, self.norb, self.nelec, c1._strs, nroots=nroots
        )
        if nroots == 1:
            self.c = c
        else:
            self.c = c[nroots - 1]
        self.myci = myci
        if nroots == 1:
            return e, c
        return e[nroots - 1], c[nroots - 1]


    def make_rdm(self):
        """
        Get density matrices from the selected CI wavefunction
                
        Returns:
            tuple[np.ndarray, np.ndarray]: one-particle density matrix, two-particle density matrix
        """
        myci = fci.select_ci.SelectedCI()
        dm1, dm2 = myci.make_rdm12(self.c, self.norb, self.nelec)
        return dm1, dm2


if __name__ == "__main__":
    norb = 2
    nelec = 2
    int1e = np.random.rand(norb, norb)
    int1e = int1e + int1e.T
    int2e = np.zeros((norb, norb, norb, norb))
    uccsd = UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.optimize()
    smp = Sampler(uccsd)
    qsci = QSCI(smp)
    e, c, _ = qsci.diagonalize()
    print(e)
    print(c)

    from pyscf import fci

    cis = fci.direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    print(e)
    print(c)
