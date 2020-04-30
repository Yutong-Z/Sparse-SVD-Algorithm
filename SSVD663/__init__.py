#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:28:59 2020

@author: yutongzhang jiaxiyin
"""

# SSVD663.SSVD

def update_u(X,u,v,s,n,d,gamma_u):
    '''
    update the left singular vector
    
    X = data matrix
    u = left singular vector from last iteration
    v = right singular vector from last iteration
    s = sigular value
    n = the number of rows in the data matrix X
    d = the number of columns in the data matrix X
    gamma_u = weight parameter
    '''
    Xv = X @ v
    w1 = np.abs(Xv) ** (-gamma_u)
    lambdas = np.sort(np.unique(np.append(np.abs(Xv / w1), 0)))[0:-1]
    BICs = []
    us = []
    sigmaS = np.sum((X - s * u @ v.T) ** 2) / (n*d - d)  
    for lambda_u in lambdas:
        u_h = np.sign(Xv) * (np.abs(Xv) >= (lambda_u * w1)) * (np.abs(Xv) - lambda_u * w1)
        BIC = np.sum((X - u_h @ v.T) ** 2) / sigmaS / n / d + np.sum(u_h != 0) * np.log( n*d ) / n / d
        BICs.append(BIC)
        us.append(u_h)
    u_new = us[BICs.index(min(BICs))]
    
    u_new = u_new / np.sqrt(np.sum(u_new ** 2))
    return u_new

def update_v(X,u,v,s,n,d,gamma_v):
    '''
    update the right singular vector
    
    X = data matrix
    u = left singular vector from last iteration
    v = right singular vector from last iteration
    s = sigular value
    n = the number of rows in the data matrix X
    d = the number of columns in the data matrix X
    gamma_v = weight parameter
    '''
    Xu = X.T @ u
    w2 = np.abs(Xu) ** (-gamma_v)
    lambdas = np.sort(np.unique(np.append(np.abs(Xu / w2), 0)))[0:-1]
    BICs = []
    vs = []
    sigmaS = np.sum((X - s * u @ v.T) ** 2) / (n*d - d)  
    for lambda_v in lambdas:
        v_h = np.sign(Xu) * (np.abs(Xu) >= (lambda_v * w2)) * (np.abs(Xu)-lambda_v * w2)
        BIC = np.sum((X-u @ v_h.T) ** 2) / sigmaS / n / d + np.sum(v_h != 0) * np.log(n * d) / n / d
        BICs.append(BIC)
        vs.append(v_h)
    v_new = vs[BICs.index(min(BICs))]
    
    v_new = v_new / np.sqrt(np.sum(v_new ** 2))
    return v_new

def SSVD(X, gamma_u = 2, gamma_v=2, tol = 1e-4):
    
    '''
    SSVD for the first layer
    
    X = data matrix
    gamma_u, gamma_v = weight parameters, default = 2
    tol = tolerance for convergence, default to 1e-4
    '''
    

    
    U, S, Vt = np.linalg.svd(X)
    u = U[:,0].reshape(-1,1)
    v = Vt[0].reshape(-1,1)
    s = S[0]
    n = X.shape[0]
    d = X.shape[1]
    
    du = 1
    dv = 1
  

    while((du > tol) or (dv > tol)):
        v_new = update_v(X,u,v,s,n,d,gamma_v)
        dv = np.sqrt(np.sum((v - v_new) ** 2))
        v = v_new

        u_new = update_u(X,u,v,s,n,d,gamma_u)
        du = np.sqrt(np.sum((u - u_new) ** 2))
        u = u_new

    s = u.T @ X @ v

    return u, s, v


# SSVD663.SSVD_multi

import numpy as np
from multiprocessing import Pool
from functools import partial

def BIC1(lambda_v, Xu, X, w2, u, sigmaS, n, d):
    '''
    Calculate BIC for lambda_v
    '''
    v_h = np.sign(Xu) * (np.abs(Xu) >= (lambda_v * w2)) * (np.abs(Xu)-lambda_v * w2)
    BIC = np.sum((X-u @ v_h.T) ** 2) / sigmaS / n / d + np.sum(v_h != 0) * np.log(n * d) / n / d
    
    return BIC

def update_v_multi(X,u,v,s,n,d,gamma_v=2):
    '''
    update the right singular vector
    
    X = data matrix
    u = left singular vector from last iteration
    v = right singular vector from last iteration
    s = sigular value
    n = the number of rows in the data matrix X
    d = the number of columns in the data matrix X
    gamma_v = weight parameter
    '''
    Xu = X.T @ u
    w2 = np.abs(Xu) ** (-gamma_v)
    lambda_v = np.sort(np.unique(np.append(np.abs(Xu / w2), 0)))[0:-1]
    sigmaS = np.sum((X - s * u @ v.T) ** 2) / (n*d - d)
    
    BIC1_partial = partial(BIC1, Xu = Xu, X = X, w2 = w2, u = u , sigmaS = sigmaS, n = n, d = d)
    with Pool(processes=4) as pool:
        BIC = pool.map(BIC1_partial, lambda_v)
    
    lambda_v_star = lambda_v[np.argmin(BIC)]
    v_new = np.sign(Xu) * (np.abs(Xu) >= (lambda_v_star * w2)) * (np.abs(Xu)-lambda_v_star * w2)
    v_new = v_new / np.sqrt(np.sum(v_new ** 2))

    return v_new

def BIC2(lambda_u, Xv, X, w1, v, sigmaS, n, d):    
    '''
    Calculate BIC for lambda_u
    '''
    u_h = np.sign(Xv) * (np.abs(Xv) >= (lambda_u * w1)) * (np.abs(Xv) - lambda_u * w1)
    BIC = np.sum((X - u_h @ v.T) ** 2) / sigmaS / n / d + np.sum(u_h != 0) * np.log( n*d ) / n / d
    
    return BIC

def update_u_multi(X,u,v,s,n,d,gamma_u=2):
    '''
    update the left singular vector
    
    X = data matrix
    u = left singular vector from last iteration
    v = right singular vector from last iteration
    s = sigular value
    n = the number of rows in the data matrix X
    d = the number of columns in the data matrix X
    gamma_u = weight parameter
    '''
    Xv = X @ v
    w1 = np.abs(Xv) ** (-gamma_u)
    lambda_u = np.sort(np.unique(np.append(np.abs(Xv / w1), 0)))[0:-1]
    sigmaS = np.sum((X - s * u @ v.T) ** 2) / (n*d - d)
    
    BIC2_partial = partial(BIC2, Xv = Xv, X = X, w1 = w1, v = v , sigmaS = sigmaS, n = n, d = d)
    with Pool(processes=4) as pool:
        BIC = pool.map(BIC2_partial, lambda_u)
    
    lambda_u_star = lambda_u[np.argmin(BIC)]
    u_new = np.sign(Xv) * (np.abs(Xv) >= (lambda_u_star * w1)) * (np.abs(Xv) - lambda_u_star * w1)
    u_new = u_new / np.sqrt(np.sum(u_new ** 2))
    return u_new

def SSVD_multi(X, tol = 1e-4):
    """
    SSVD for the first layer
    
    X = data matrix
    tol = tolerance for convergence, default to 1e-4
    
    """
    U, S, Vt = np.linalg.svd(X)
    u = U[:,0].reshape(-1,1)
    v = Vt[0].reshape(-1,1)
    s = S[0]
    n = X.shape[0]
    d = X.shape[1]
    
    du = 1
    dv = 1
    count = 0

    while((du > tol) or (dv > tol)):
        v_new = update_v_multi(X,u,v,s,n,d,gamma_v=2)
        dv = np.sqrt(np.sum((v - v_new) ** 2))
        v = v_new

        u_new = update_u_multi(X,u,v,s,n,d,gamma_u=2)
        du = np.sqrt(np.sum((u - u_new) ** 2))
        u = u_new
        
        count = count + 1
        
    s = u.T @ X @ v
    
    return u, s, v
