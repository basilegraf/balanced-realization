#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:02:19 2021

Balanced realization from Markov parameters

@author: basile
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class LTISystem():
    def __init__(self, A, B ,C , D):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)
        self.D = np.asarray(D)
        
    def markov(self, n):
        if n==0:
            return self.D
        else:
            return self.C @ np.linalg.matrix_power(self.A, n - 1) @ self.B
        
    def hankle(self, n0, r):
        p = self.B.shape[1]
        q = self.C.shape[0]
        t = list(map(self.markov, list(range(n0, n0 + 2*r -1))))
        T = np.zeros((r * q, r * p))
        for i in range(r):
            for j in range(r):
                T[i * q : (i + 1) * q, j * p : (j + 1) * p] = t[i + j]
        return T
        
    def hankelT(self):
        n = self.A.shape[0]
        return self.hankle(1, n)
    
    def hankelTTilde(self):
        n = self.A.shape[0]
        return self.hankle(2, n)
    
    def balanced(self, tol = 1.0e-13):
        p = self.B.shape[1]
        q = self.C.shape[0]
        T = self.hankelT()
        Tt = self.hankelTTilde()
        (u,s,vh) = np.linalg.svd(T, full_matrices = False)
        rank = (s/(s[0]+1) > tol).sum()
        if rank > self.A.shape[0]:
            rank = self.A.shape[0]
        u = u[:,0:rank]
        s = s[0:rank]
        vh = vh[0:rank, :]
        obs = u @ np.diag(np.sqrt(s))
        cont = np.diag(np.sqrt(s)) @ vh
        AA = np.linalg.pinv(obs) @ Tt @ np.linalg.pinv(cont)
        BB = cont[:, 0:p]
        CC = obs[0:q, :]
        DD = self.D
        return LTISystem(AA, BB, CC, DD)


A = 0.1 * np.asarray([[0,2,1], [0,-2,3], [1,-4,2]])   
B = 100 * np.asarray([[0.2,0], [0, 1.1], [-2,0.4]])
C = 0.01 * np.asarray([[0.7,-0.3,0], [0, 1.1, 3.2]])
D = np.asarray([[-0.2,0], [.2, .1]])

sys = LTISystem(A,B,C,D)
sysb = sys.balanced()

for k in range(0):
    ma = sys.markov(k)
    mb = sysb.markov(k)
    err = np.linalg.norm(ma - mb) / (np.linalg.norm(ma) + 1)
    print(err)
