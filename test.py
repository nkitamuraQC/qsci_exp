import pytest
from qsci import qsciclass, vqe
from pyscf.fci import direct_spin1, FCI
import numpy as np
from pyscf import gto, scf, ao2mo


def test_energy():
    norb = 2
    nelec = 2
    int1e = np.random.rand(norb, norb)
    int1e = int1e + int1e.T
    int2e = np.zeros((norb, norb, norb, norb))
    uccsd = vqe.UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.optimize()
    smp = qsciclass.Sampler(uccsd)
    qscicls = qsciclass.QSCI(smp)
    qscicls.nchoose = 4
    e1, c1 = qscicls.diagonalize_sci()
    cis = direct_spin1.FCISolver()
    e2, c2 = cis.kernel(int1e, int2e, norb, nelec)
    assert abs(e1 - e2) < 1e-6
    return


def test_H2():
    norb = 2
    nelec = 2
    mol = gto.Mole()
    mol.atom = """H 0 0 0; H 0 0 1"""
    mol.basis = "sto-3g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    hcore = mf.get_hcore()
    int1e = np.einsum("ia,jb,ij->ab", mf.mo_coeff, mf.mo_coeff, hcore)
    int2e = ao2mo.kernel(mol, mf.mo_coeff)
    int2e = ao2mo.addons.restore("s1", int2e, mf.mo_coeff.shape[1])
    print(int2e.shape)
    uccsd = vqe.UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.optimize()
    smp = qsciclass.Sampler(uccsd)
    qscicls = qsciclass.QSCI(smp)
    qscicls.nchoose = 4
    e1, c1 = qscicls.diagonalize_sci()
    cis = direct_spin1.FCISolver()
    e2, c2 = cis.kernel(int1e, int2e, norb, nelec)
    assert abs(e1 - e2) < 1e-6
    return
