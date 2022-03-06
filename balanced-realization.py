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




# Kronecker delta funtion
def delta(p,q):
    if p==q:
        return 1.0
    else:
        return 0.0
    
    
def matrixMContinuous(A):
    A = np.asarray(A)
    assert len(A.shape) == 2, "A should be 2D"
    assert A.shape[0] == A.shape[1], "A should be square"
    n = A.shape[0]
    M = np.zeros((n**2, n**2))
    for s in range(n**2):
        for l in range(n**2):
            M[s,l] = A[s//n, l//n] * delta(l%n, s%n) + A[s%n, l%n]  * delta(l//n, s//n)
    return M

def matrixMDiscrete(A):
    A = np.asarray(A)
    assert len(A.shape) == 2, "A should be 2D"
    assert A.shape[0] == A.shape[1], "A should be square"
    n = A.shape[0]
    M = np.zeros((n**2, n**2))
    for s in range(n**2):
        for l in range(n**2):
            M[s,l] = A[s//n, l//n] * A[s%n, l%n] - delta(l, s) #delta(l%n, s%n) * delta(l//n, s//n)
    return M

def vectorb(BBt):
    BBt = np.asarray(BBt)
    assert len(BBt.shape) == 2, "BBt should be 2D"
    assert BBt.shape[0] == BBt.shape[1], "BBt should be square"
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

# Continuous and discrete versions differ only in the form of the size n**2
# square system 
def ControllabilityGramian(M, A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    BBt = np.matmul(B, B.transpose())
    # n**2 x n**2 linear system
    b = vectorb(BBt)   
    w = np.linalg.solve(M,-b)
    return vector2matrix(w)

# Solve for continuous controllability gramian A*W + W*A' + B*B'
def ControllabilityGramianContinuous(A, B):
    M = matrixMContinuous(A)
    return ControllabilityGramian(M, A, B)

# Solve for discrete controllability gramian A*W*A' - W + B*B'
def ControllabilityGramianDiscrete(A, B):
    M = matrixMDiscrete(A)
    return ControllabilityGramian(M, A, B)

# Solve for continuous observability gramian A'*M + M*A + C'*C
def ObservabilityGramianContinuous(A, C):
    A = np.asarray(A)
    C = np.asarray(C)
    return ControllabilityGramianContinuous(A.transpose(), C.transpose())

# Solve for discrete observability gramian A'*W*A - W + C'*C
def ObservabilityGramianDiscrete(A, C):
    A = np.asarray(A)
    C = np.asarray(C)
    return ControllabilityGramianDiscrete(A.transpose(), C.transpose())


def BalancedRealization(Wc,Wo,A,B,C,D=None):
    # Wc = R'*R  (Wc is symmetric and pos def)
    R = np.linalg.cholesky(Wc).transpose()
    # Compute decomposition R*Wo*R' = U*S*S*U'
    # Symetric pos def => use eig val and eig vec
    RWoRt = np.matmul(np.matmul(R, Wo), R.transpose())
    ev, U = np.linalg.eig(RWoRt)
    S = np.diag(np.sqrt(ev))
    # Get similarity transform P = sqS * U' * inv(R')
    sqS = np.diag(np.sqrt(np.sqrt(ev))) # Square root of S
    P = np.matmul(np.matmul(sqS, U.transpose()), np.linalg.inv(R.transpose()))
    Abar = np.matmul(np.matmul(P, A), np.linalg.inv(P))
    Bbar = np.matmul(P, B)
    Cbar = np.matmul(C, np.linalg.inv(P))
    Dbar = D
    return Abar, Bbar, Cbar, Dbar, P
    

def BalancedRealizationContinuous(A,B,C,D=None):
    Wc = ControllabilityGramianContinuous(A, B)
    Wo = ObservabilityGramianContinuous(A, C)
    return BalancedRealization(Wc,Wo,A,B,C,D)


def BalancedRealizationDiscrete(A,B,C,D=None):
    Wc = ControllabilityGramianDiscrete(A, B)
    Wo = ObservabilityGramianDiscrete(A, C)
    return BalancedRealization(Wc,Wo,A,B,C,D)



if __name__ == "__main__":
    import unittest
    
    # Build a stable matrix with ev inside unit circle
    def randomDiscreteSytem(n, m=1, p=1):
        A = np.random.rand(n, n)
        ev = np.linalg.eigvals(A)
        maxev = np.max(np.abs(ev))
        A *= 0.8 / maxev
        B = np.random.rand(n, m)
        C = np.random.rand(p, n)
        D = np.random.rand(p, m)
        return A,B,C,D
    
    # Build a stable matrix with ev with negative real part
    def randomContinuous(n, m=1, p=1):
        A,B,C,D = randomDiscreteSytem(n, m, p)
        A = A - 2.0*np.identity(n)
        M = np.random.rand(n, n)
        A = np.matmul(np.matmul(M,A), np.linalg.inv(M))
        return A,B,C,D
    
    def isDiagonal(M, tol = 1e-14):
        M = np.asarray(M)
        M = M - np.diag(np.diagonal(M))
        z = np.abs(M) < tol
        return np.alltrue(z)
    
    def isZero(M, tol = 1e-14):
        M = np.asarray(M)
        z = np.abs(M) < tol
        return np.alltrue(z)
    
    class TestBalancedRealization(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.n = 3 # state space dim
            cls.m = 2 # input dim
            cls.p = 2 # output dim
            cls.tol = 1.0e-12
            
        def test_ControllabilityGramianContinuous(self):
            A,B,C,D = randomContinuous(self.n, self.m, self.p)
            Wc = ControllabilityGramianContinuous(A, B)
            BBt = np.matmul(B, B.transpose())
            # check gramian equation
            errWc = np.matmul(A,Wc) + np.matmul(Wc,A.transpose()) + BBt
            relErr = np.linalg.norm(errWc) / np.linalg.norm(Wc)
            self.assertLess(relErr, self.tol, "Continuous time controllability gramian equation not satisfied")
            
        def test_ObservabilityGramianContinuous(self):
            A,B,C,D = randomContinuous(self.n, self.m, self.p)
            Wo = ObservabilityGramianContinuous(A, C)
            CtC = np.matmul(C.transpose(), C)
            # check gramian equation
            errWo = np.matmul(A.transpose(),Wo) + np.matmul(Wo,A) + CtC
            relErr = np.linalg.norm(errWo) / np.linalg.norm(Wo)
            self.assertLess(relErr, self.tol, "Continuous time observability gramian equation not satisfied")  
            
        def test_ControllabilityGramianDiscrete(self):
            A,B,C,D = randomDiscreteSytem(self.n, self.m, self.p)
            Wc = ControllabilityGramianDiscrete(A, B)
            BBt = np.matmul(B, B.transpose())
            # check gramian equation
            errWc = np.matmul(np.matmul(A,Wc),A.transpose()) - Wc + BBt
            relErr = np.linalg.norm(errWc) / np.linalg.norm(Wc)
            self.assertLess(relErr, self.tol, "Discrete time controllability gramian equation not satisfied")
            
        def test_ObservabilityGramianDiscrete(self):
            A,B,C,D = randomDiscreteSytem(self.n, self.m, self.p)
            Wo = ObservabilityGramianDiscrete(A, C)
            CtC = np.matmul(C.transpose(), C)
            # check gramian equation
            errWo = np.matmul(np.matmul(A.transpose(),Wo),A) - Wo + CtC
            relErr = np.linalg.norm(errWo) / np.linalg.norm(Wo)
            self.assertLess(relErr, self.tol, "Discrete time observability gramian equation not satisfied")  
    
        def test_BalancedRealizationContinuous(self):
            A,B,C,D = randomContinuous(self.n, self.m, self.p)
            Abar, Bbar, Cbar, Dbar, P = BalancedRealizationContinuous(A,B,C,D)
            WcBar = ControllabilityGramianContinuous(Abar, Bbar)
            WoBar = ObservabilityGramianContinuous(Abar, Cbar)
            self.assertTrue(isDiagonal(WcBar, self.tol), "Balanced realization continuous: non diagonal gramian WcBar")
            self.assertTrue(isDiagonal(WoBar, self.tol), "Balanced realization continuous: non diagonal gramian WoBar")
            self.assertTrue(isZero(WcBar-WoBar, self.tol), "Balanced realization continuous: gramian are not equal")
            
        def test_BalancedRealizationDiscrete(self):
            A,B,C,D = randomDiscreteSytem(self.n, self.m, self.p)
            Abar, Bbar, Cbar, Dbar, P = BalancedRealizationDiscrete(A,B,C,D)
            WcBar = ControllabilityGramianDiscrete(Abar, Bbar)
            WoBar = ObservabilityGramianDiscrete(Abar, Cbar)
            self.assertTrue(isDiagonal(WcBar, self.tol), "Balanced realization discrete: non diagonal gramian WcBar")
            self.assertTrue(isDiagonal(WoBar, self.tol), "Balanced realization discrete: non diagonal gramian WoBar")
            self.assertTrue(isZero(WcBar-WoBar, self.tol), "Balanced realization discrete: gramian are not equal")
    
    unittest.main()
    
    
    
