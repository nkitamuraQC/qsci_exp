# QSCI
AbinitioTools is a Python package for convinient tools of quantum chemical calculations.

## Features
- Correlation functions
  - A same-time and same-position current-current correlation function
  - A spin-spin correlation function
  - A charge-charge correlation function
  - An exciton correlation function
- Calculation of Green's function for a mean-field calculation
- Detection of a metal-insulator transition
  - Under consideration

## Usages

```python
from pyscf import gto, dft
import numpy as np
dist = 0.7
E = 10
hydrogen = gto.M(
    atom = f'''
        H  0.000000  0.00000  0.000000
        H  0.000000  0.00000  {dist}
        H  0.000000  0.00000  {dist*2}
        H  0.000000  0.00000  {dist*3}
    ''',
    basis = 'sto-3g',  # 基底関数系: STO-3Gを使用
    verbose = 0,
)
    
Efield = np.array([0, 0, E])
mf_jj = DFT_JJ(hydrogen)
mf_jj.run_dft(E)
mf_jj.calc_jj(0, 1)
```

## Installation

```shell
conda create -n abtool python=3.10
conda activate abtool
git clone https://github.com/nkitamuraQC/abinitioTools.git
cd abinitioTools
pip install -e .
```
