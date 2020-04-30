# Sparse-SVD-Algorithm
2020 Spring STA 663 Final Project 

Authors: Yutong Zhang, Jiaxi Yin

#### Implementation of Sparse Singular Value Decomposition based on "Biclustering via Sparse Singular Value Decomposition" written by Mihee Lee, Haipeng Shen, Jianhua Z. Huang, and J. S. Marron.

Package should be installed by `pip install SSVD663`

Load functions using `from SSVD663 import SSVD` and `from SSVD663 import SSVD_multi`

This package is used to find the best checkerboard structured matrix approximation to the data matrix. It includes two functions `SSVD` and `SSSVD_multi`.

`SSSVD_multi` is the optimization version of `SSSVD`, with multiprocessing method.

`SSVD` and `SSSVD_multi` both find the first layer through SSVD algorithm. The first one is an implementation of straightforward python and the second one is an optimization version with parallelism. 

### SSVD:

u, s, v = SSVD663.SSVD(X, gamma_u = 2, gamma_v=2, tol = 1e-4)

Inputs:

X = data matrix

gamma_u, gamma_v = weight parameters, default = 2

tol = tolerance for convergence, default to 1e-4

Outputs:

u = left singular vector of SSVD

s = singular value of SSVD

v = right singular vector of SSVD

### SSVD_multi

u, s, v = SSVD663.SSVD_multi(X, gamma_u = 2, gamma_v=2, tol = 1e-4)

Inputs:

X = data matrix

tol = tolerance for convergence, default to 1e-4

Outputs:

u = left singular vector of SSVD

s = singular value of SSVD

v = right singular vector of SSVD
