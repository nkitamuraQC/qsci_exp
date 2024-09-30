import pytest
from abtools.correlation import AbinitioToolsclass
from pyscf import gto
from pyscf.fci import direct_spin1

def generate_ints():
    return

def test_spin_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints()
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_spin_corr(0, 0)
    s01 = myclass.calc_spin_corr(0, 1)
    s02 = myclass.calc_spin_corr(0, 2)
    assert(s00 > s01)
    assert(s01 > s02)
    return

def test_chg_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints()
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_chg_corr(0, 0)
    s01 = myclass.calc_chg_corr(0, 1)
    s02 = myclass.calc_chg_corr(0, 2)
    assert(s00 > s01)
    assert(s01 > s02)
    return


def test_cc_corr():
    norb = 8
    nelec = 12
    mol = gto.Mole()
    myclass = AbinitioToolsclass(mol)
    int1e, int2e = generate_ints()
    cis = direct_spin1.FCISolver()
    e, c = cis.kernel(int1e, int2e, norb, nelec)
    dm1, dm2 = cis.make_rdm12s(c, norb, nelec)
    myclass.dm1 = dm1
    myclass.dm2 = dm2
    s00 = myclass.calc_jj(0, 0)
    s01 = myclass.calc_jj(0, 1)
    s02 = myclass.calc_jj(0, 2)
    assert(s00 > s01)
    assert(s01 > s02)
    return
   