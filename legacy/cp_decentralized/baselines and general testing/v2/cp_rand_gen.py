#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:07:54 2020

@author: junxuanli

Random event generator (for parallel comparison) -- two facilities network
    1. Demand generator
    2. Suppliler state generator
"""

import numpy as np
from scipy.stats import poisson
from scipy.stats import bernoulli

# In[]:
def gen_D(d,T,N):
    # d: demand rate
    D = {}
    D[0] = poisson.rvs(d[0],size=(N,T))
    D[1] = poisson.rvs(d[1],size=(N,T))
    return D

# In[]:
def gen_A(p,T,N,cor): 
    # p: prob. of disruption (different from A_sample_ind)
    # cor: 'i' - independent, 'ii' - identical, 'iii' - flipped
    A = {}
    pm1 = np.zeros((N,T)) + p[0]
    pm2 = np.zeros((N,T)) + p[1]
    A1 = bernoulli.rvs(pm1)
    A2 = bernoulli.rvs(pm2)
    if cor == 'i':
        A[0] = A1
        A[1] = A2
    if cor == 'ii':
        A[0] = A1
        A[1] = np.copy(A1)
    if cor == 'iii':
        A[0] = A1
        A[1] = 1 - A1
    return A