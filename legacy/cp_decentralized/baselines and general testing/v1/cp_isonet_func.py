#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:44:48 2020

@author: junxuanli

Iso-Net functions -- two facilities network
"""

import numpy as np
from scipy.stats import poisson
import copy

# In[]:
def get_Zdict(d,p,paras,beta):
    cR, hR, pR = paras[0], paras[1], paras[2]
    rhoR = (pR-(1-beta)*cR)/(pR+hR)
    Zdict = {}
    for j in range(2):
        Ztemp = np.zeros(30)
        z = poisson.ppf(rhoR,d[j])
        for k in range(30):
            wv = np.array([p[j]**i*beta**i for i in range(k+1)])
            wv[0] = 1
            funv = np.array([-pR+(pR+hR)*poisson.cdf(z,d[j]*(i+1)) for i in range(k+1)])
            phi = (1-beta)*cR+np.dot(wv,funv)
            while phi<0:
                z = z+1
                funv = np.array([-pR+(pR+hR)*poisson.cdf(z,d[j]*(i+1)) for i in range(k+1)])
                phi = (1-beta)*cR+np.dot(wv,funv)
            Ztemp[k] = z
        Zdict[j] = np.copy(Ztemp)
    return Zdict

# In[]:
def get_a_iso(x,Zdict,At,t,T):
    a = np.zeros(2)
    for i in range(2):
        if At[i] == 0:
            Z = Zdict[i]
            a[i] = max([0,x[i]['s']-x[i]['r']+Z[min([29,T-t])]])
    return a

# In[]:   
def sim_iso(d,L,T,x,Dv,Av,B,Zdict):
    sim_res = {}
    
    Bv1 = np.zeros(T)
    sv1 = np.zeros(T)
    b0v1 = np.zeros(T)
    rv1 = np.zeros(T)
        
    av1 = np.zeros(T)
    qv1 = np.zeros(T)
    qv1[0] = B[0]
        
    bsv1 = np.zeros(T)
    rsv1 = np.zeros(T)
    bov1 = np.zeros(T)
    rov1 = np.zeros(T)
    
    Bv2 = np.zeros(T)
    sv2 = np.zeros(T)
    b0v2 = np.zeros(T)
    rv2 = np.zeros(T)
        
    av2 = np.zeros(T)
    qv2 = np.zeros(T)
    qv2[0] = B[1]
        
    bsv2 = np.zeros(T)
    rsv2 = np.zeros(T)
    bov2 = np.zeros(T)
    rov2 = np.zeros(T)
    
    t = 0
    while t<T:
        Bv1[t] = np.sum(x[0]['b'])
        Bv2[t] = np.sum(x[1]['b'])
        sv1[t] = x[0]['s']
        sv2[t] = x[1]['s']
        b0v1[t] = x[0]['b'][0]
        b0v2[t] = x[1]['b'][0]
        rv1[t] = x[0]['r']
        rv2[t] = x[1]['r']
        
        a = get_a_iso(x, Zdict, [Av[0][t],Av[1][t]], t, T)
        av1[t] = a[0]
        av2[t] = a[1]
        
        m1 = np.min([x[0]['s'],x[0]['b'][0],x[0]['r']])
        m2 = np.min([x[1]['s'],x[1]['b'][0],x[1]['r']])
        x[0]['s'] = x[0]['s']-m1+Dv[0][t]
        x[1]['s'] = x[1]['s']-m2+Dv[1][t]
        x[0]['b'][0] = x[0]['b'][0]-m1+qv1[t]+x[0]['b'][1]
        x[1]['b'][0] = x[1]['b'][0]-m2+qv2[t]+x[1]['b'][1]
        for i in np.arange(1,L-1):
            x[0]['b'][i] = x[0]['b'][i+1]
            x[1]['b'][i] = x[1]['b'][i+1]
        x[0]['b'][L-1] = m1
        x[1]['b'][L-1] = m2
        x[0]['r'] = x[0]['r']-m1+a[0]
        x[1]['r'] = x[1]['r']-m2+a[1]
        
        bsv1[t] = np.max([x[0]['s']-x[0]['b'][0],0])
        bsv2[t] = np.max([x[1]['s']-x[1]['b'][0],0])
        rsv1[t] = np.max([x[0]['s']-x[0]['r'],0])
        rsv2[t] = np.max([x[1]['s']-x[1]['r'],0])
        
        bov1[t] = np.max([-x[0]['s']+x[0]['b'][0],0])
        bov2[t] = np.max([-x[1]['s']+x[1]['b'][0],0])
        rov1[t] = np.max([-x[0]['s']+x[0]['r'],0])
        rov2[t] = np.max([-x[1]['s']+x[1]['r'],0])
        
        t = t+1
    
    sim_res[0] = {'B':Bv1,'s':sv1,'b0':b0v1,'r':rv1,'a':av1,'q':qv1,'bs':bsv1,'rs':rsv1,'bo':bov1,'ro':rov1}
    sim_res[1] = {'B':Bv2,'s':sv2,'b0':b0v2,'r':rv2,'a':av2,'q':qv2,'bs':bsv2,'rs':rsv2,'bo':bov2,'ro':rov2}

    return sim_res

# In[]:
def test_iso(d,L,x,D,A,B,paras,p,beta):
    test_res = {}
    
    N = np.shape(D[0])[0]
    T = np.shape(D[0])[1]
    Bm1 = np.zeros((N,T))
    sm1 = np.zeros((N,T))
    b0m1 = np.zeros((N,T))
    rm1 = np.zeros((N,T))
    
    am1 = np.zeros((N,T))
    qm1 = np.zeros((N,T))
    
    bsm1 = np.zeros((N,T))
    rsm1 = np.zeros((N,T))
    bom1 = np.zeros((N,T))
    rom1 = np.zeros((N,T))
    
    Bm2 = np.zeros((N,T))
    sm2 = np.zeros((N,T))
    b0m2 = np.zeros((N,T))
    rm2 = np.zeros((N,T))
    
    am2 = np.zeros((N,T))
    qm2 = np.zeros((N,T))
    
    bsm2 = np.zeros((N,T))
    rsm2 = np.zeros((N,T))
    bom2 = np.zeros((N,T))
    rom2 = np.zeros((N,T))
    
    Zdict = get_Zdict(d, p, paras, beta)
    for n in range(N):
        xtemp = copy.deepcopy(x)
        sim_res = sim_iso(d, L, T, xtemp, {0:D[0][n],1:D[1][n]}, {0:A[0][n],1:A[1][n]}, B, Zdict)
        Bm1[n], sm1[n], b0m1[n], rm1[n], am1[n], qm1[n], bsm1[n], rsm1[n], bom1[n], rom1[n] = sim_res[0]['B'], sim_res[0]['s'], sim_res[0]['b0'],sim_res[0]['r'],sim_res[0]['a'],sim_res[0]['q'],sim_res[0]['bs'],sim_res[0]['rs'],sim_res[0]['bo'],sim_res[0]['ro']
        Bm2[n], sm2[n], b0m2[n], rm2[n], am2[n], qm2[n], bsm2[n], rsm2[n], bom2[n], rom2[n] = sim_res[1]['B'], sim_res[1]['s'], sim_res[1]['b0'],sim_res[1]['r'],sim_res[1]['a'],sim_res[1]['q'],sim_res[1]['bs'],sim_res[1]['rs'],sim_res[1]['bo'],sim_res[1]['ro']
    
    test_res[0] = {'B':Bm1,'s':sm1,'b0':b0m1,'r':rm1,'a':am1,'q':qm1,'bs':bsm1,'rs':rsm1,'bo':bom1,'ro':rom1}    
    test_res[1] = {'B':Bm2,'s':sm2,'b0':b0m2,'r':rm2,'a':am2,'q':qm2,'bs':bsm2,'rs':rsm2,'bo':bom2,'ro':rom2}
    
    # print(x)
    
    return test_res   

# In[]:
def cost_iso(test_res,paras,beta):
    cR, hR, pR, cB, hB, pB = paras[0], paras[1], paras[2], paras[3], paras[4], paras[5]
    am1, qm1, bom1, rom1, bsm1, rsm1 = test_res[0]['a'], test_res[0]['q'], test_res[0]['bo'], test_res[0]['ro'],test_res[0]['bs'], test_res[0]['rs']
    am2, qm2, bom2, rom2, bsm2, rsm2 = test_res[1]['a'], test_res[1]['q'], test_res[1]['bo'], test_res[1]['ro'],test_res[1]['bs'], test_res[1]['rs']
    N = np.shape(am1)[0]
    T = np.shape(am1)[1]
    betaT =np.matrix([beta**t for t in range(T)])
    oneN = np.transpose(np.matrix(np.ones(N)))
    betam = np.dot(oneN,betaT)
    
    costm1 = cB*qm1+cR*am1+hB*bom1+hR*rom1+pB*bsm1+pR*rsm1
    costdm1 = np.multiply(betam,costm1)
    costdv1 = np.sum(costdm1,axis=1)
    ec1 = np.sum(costdv1)/N
    
    costm2 = cB*qm2+cR*am2+hB*bom2+hR*rom2+pB*bsm2+pR*rsm2
    costdm2 = np.multiply(betam,costm2)
    costdv2 = np.sum(costdm2,axis=1)
    ec2 = np.sum(costdv2)/N
    return ec1,np.squeeze(np.asarray(costdv1)),ec2,np.squeeze(np.asarray(costdv2))

# In[]:
def solve_iso(d,L,x,Dtr,Atr,paras,p,beta):
    cB, hB, pB = paras[3], paras[4], paras[5]
    rhoB = (pB-(1-beta)*cB)/(pB+hB)
    # facility 1
    # print('F1 OPT...')
    q1 = max([0,poisson.ppf(rhoB,L*d[0])-x[0]['b'][0]])
    q2 = q1+1
    test_res1 = test_iso(d, L, x, Dtr, Atr, [q1,q1], paras, p, beta)
    test_res2 = test_iso(d, L, x, Dtr, Atr, [q2,q2], paras, p, beta)
    ec1, _, _, _ = cost_iso(test_res1, paras, beta)
    ec2, _, _, _ = cost_iso(test_res2, paras, beta)
    delta = ec2-ec1
    # print(delta)
    if delta == 0:
        B1 = q1
    if delta < 0:
        while delta < 0:
            ec1 = ec2
            q1 = q2
            q2 = q2+1
            test_res2 = test_iso(d, L, x, Dtr, Atr, [q2,0], paras, p, beta)
            ec2, _, _, _ = cost_iso(test_res2, paras, beta)
            delta = ec2-ec1
            print(delta)
        B1 = q1
    if delta > 0:
        while delta > 0:
            ec2 = ec1
            q2 = q1
            q1 = q1-1
            test_res1 = test_iso(d, L, x, Dtr, Atr, [q1,0], paras, p, beta)
            ec1, _, _, _ = cost_iso(test_res1, paras, beta)
            delta = ec2-ec1
            print(delta)
        B1 = q2
    # facility 2
    print('F2 OPT...')
    q1 = max([0,poisson.ppf(rhoB,L*d[1])-x[1]['b'][0]])
    q2 = q1+1
    test_res1 = test_iso(d, L, x, Dtr, Atr, [B1,q1], paras, p, beta)
    test_res2 = test_iso(d, L, x, Dtr, Atr, [B1,q2], paras, p, beta)
    _, _, ec1, _ = cost_iso(test_res1, paras, beta)
    _, _, ec2, _ = cost_iso(test_res2, paras, beta)
    delta = ec2-ec1
    print(delta)
    if delta == 0:
        B2 = q1
    if delta < 0:
        while delta < 0:
            ec1 = ec2
            q1 = q2
            q2 = q2+1
            test_res2 = test_iso(d, L, x, Dtr, Atr, [B1,q2], paras, p, beta)
            _, _, ec2, _ = cost_iso(test_res2, paras, beta)
            delta = ec2-ec1
            print(delta)
        B2 = q1
    if delta > 0:
        while delta > 0:
            ec2 = ec1
            q2 = q1
            q1 = q1-1
            test_res1 = test_iso(d, L, x, Dtr, Atr, [B1,q1], paras, p, beta)
            _, _, ec1, _ = cost_iso(test_res1, paras, beta)
            delta = ec2-ec1
            print(delta)
        B2 = q2
    return [B1,B2]
