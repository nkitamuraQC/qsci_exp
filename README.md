# QSCI
QSCI is a Python package for quantum computing related to quantum chemical calculations.

## Features
- Selected CI for quantum computing
  - see https://arxiv.org/abs/2302.11320

## Usages

```python
from qsci import qsci, sampling
norb = 2
nelec = 2
int1e = np.random.rand(norb, norb)
int1e = int1e + int1e.T
int2e = np.zeros((norb, norb, norb, norb))
uccsd = sampling.UCCSD_Lattice(int1e, int2e, norb, nelec)
uccsd.optimize()
smp = qsci.Sampler(uccsd)
qscicls = qsci.QSCI(smp)
e, c, _ = qscicls.diagonalize()
print(e)
print(c)

from pyscf import fci

cis = fci.direct_spin1.FCISolver()
e, c = cis.kernel(int1e, int2e, norb, nelec)
print(e)
print(c)
```

## Installation

```shell
conda create -n qsci python=3.9
conda activate qsci
git clone https://github.com/nkitamuraQC/QSCI.git
cd QSCI
pip install -e .
```