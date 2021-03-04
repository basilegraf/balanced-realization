#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:02:19 2021

compute controllability and observability gramians by setting up and solving
the corresponding size n**2 linear set of equations

@author: basile
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def delta(p,q):
    if p==q:
        return 1.0
    else:
        return 0.0
    
    
def matrixMcont(A):
    A = np.asarray(A)
    assert len(A.shape) == 2, "A should be 2D"
    assert A.shape[0] == A.shape[1], "A should be square"
    n = A.shape[0]
    M = np.zeros((n**2, n**2))
    for s in range(n**2):
        for l in range(n**2):
            M[s,l] = A[s//n, l//n] * delta(l%n, s%n) + A[s%n, l%n]  * delta(l//n, s//n)
    return M

def matrixMdiscr(A):
    A = np.asarray(A)
    assert len(A.shape) == 2, "A should be 2D"
    assert A.shape[0] == A.shape[1], "A should be square"
    n = A.shape[0]
    M = np.zeros((n**2, n**2))
    for s in range(n**2):
        for l in range(n**2):
            M[s,l] = A[s//n, l//n] * A[s%n, l%n] - delta(l%n, s%n) * delta(l//n, s//n)
    return M

def vectorb(BBt):
    BBt = np.asarray(BBt)
    assert len(BBt.shape) == 2, "BBt should be 2D"
    assert B.shape[0] == BBt.shape[1], "BBt should be square"
    n = BBt.shape[0]
    b = np.zeros((n**2))
    for s in range(n**2):
        b[s] = BBt[s//n, s%n]
    return b

def vector2matrix(w):
    w = np.asarray(w)
    assert len(w.shape) == 1, "Input should be 1D"
    n = int(np.sqrt(len(w)))
    assert len(w) == n**2, "Vector length not a square"
    W = np.zeros((n, n))
    for q in range(n):
        for p in range(n):
            W[q,p] = w[n*p + q]
    return W


# Solve for continuous controllability gramian A*W + W*A' + B*B'
def ControllabilityGramianConinuous(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    BBt = np.matmul(B, B.transpose())
    # n**2 x n**2 linear system
    M = matrixMcont(A)
    b = vectorb(BBt)   
    w = np.linalg.solve(M,-b)
    return vector2matrix(w)

# Solve for discrete controllability gramian A*W*A' - W + B*B'
def ControllabilityGramianDiscrete(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    BBt = np.matmul(B, B.transpose())
    # n**2 x n**2 linear system
    M = matrixMdiscr(A)
    b = vectorb(BBt)   
    w = np.linalg.solve(M,-b)
    return vector2matrix(w)

# Solve for continuous observability gramian A'*M + M*A + C'*C
def ObservabilityGramianConinuous(A, C):
    A = np.asarray(A)
    C = np.asarray(C)
    return ControllabilityGramianConinuous(A.transpose(), C.transpose())

# Solve for discrete observability gramian A'*W*A - W + C'*C
def ObservabilityGramianDiscrete(A, C):
    A = np.asarray(A)
    C = np.asarray(C)
    return ControllabilityGramianDiscrete(A.transpose(), C.transpose())

if __name__ == "__main__":
    
    n = 3
    A = np.random.rand(n,n)/10
    B = np.random.rand(n,1)
    C = np.random.rand(1,n)
    BBt = np.matmul(B, B.transpose())
    CtC = np.matmul(C.transpose(), C)
        
    Wc = ControllabilityGramianConinuous(A, B)
    Mc = ObservabilityGramianConinuous(A, C)
    
    Wd = ControllabilityGramianDiscrete(A, B)
    Md = ObservabilityGramianDiscrete(A, C)
    
    errWc = np.matmul(A,Wc) + np.matmul(Wc,A.transpose()) + BBt
    print(errWc)
    
    errWd = np.matmul(np.matmul(A,Wd),A.transpose()) - Wd + BBt
    print(errWd)
    
    errMc = np.matmul(A.transpose(),Mc) + np.matmul(Mc,A) + CtC
    print(errMc)
    
    errMd = np.matmul(np.matmul(A.transpose(),Md),A) - Md + CtC
    print(errWd)