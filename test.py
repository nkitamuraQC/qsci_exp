import pytest
from qsci import qsciclass, vqe
from pyscf.fci import direct_spin1
import numpy as np


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
