#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:41:33 2020

@author: junxuanli

Functions used for decentralized CP under supplier disruption risks

Call functions from this file
"""

import numpy as np
from scipy.stats import poisson
from scipy.stats import bernoulli
import gurobipy as gp
from gurobipy import GRB

# In[]:
def d_sample(d,T,N):
    D = poisson.rvs(d,size=(N,T),random_state=0)
    return D

def A_sample_cor(p,T,N,cor): 
    # p: prob. of disruption (different from A_sample_ind)
    # cor: 0 - independent, 1 - identical, -1 - flipped
    A = {}
    pm = np.zeros((N,T)) + p
    A1 = bernoulli.rvs(pm)
    A2 = bernoulli.rvs(pm)
    if cor == 0:
        A[0] = np.copy(A1)
        A[1] = np.copy(A2)
    elif cor == 1:
        A[0] = np.copy(A1)
        A[1] = np.copy(A1)
    else:
        A[0] = np.copy(A1)
    return A

# In[]: Iso-Net functions
def get_Z_p(d,cR,hR,pR,beta,p):
    Zdict = np.zeros(30)
    rhoR = (pR-(1-beta)*cR)/(pR+hR)
    z = poisson.ppf(rhoR,d)
    for k in range(30):
        wv = np.array([p**i*beta**i for i in range(k+1)])
        wv[0] = 1
        funv = np.array([-pR+(pR+hR)*poisson.cdf(z,d*(i+1)) for i in range(k+1)])
        phi = (1-beta)*cR+np.dot(wv,funv)
        while phi<0:
            z = z+1
            funv = np.array([-pR+(pR+hR)*poisson.cdf(z,d*(i+1)) for i in range(k+1)])
            phi = (1-beta)*cR+np.dot(wv,funv)
        Zdict[k] = z
    return Zdict

def get_a_iso(s,r,Zdict,A,t,T):
    if A==1:
        a = 0
    else:
        H = min([29,T-t])    
        y = Zdict[H]+s        
        a = max([0,y-r])
    return a

def cost_eval_iso(cB,cR,hB,hR,pB,pR,beta,am,qm,bom,rom,bsm,rsm):
    N = np.shape(am)[0]
    T = np.shape(am)[1]
    betaT =np.matrix([beta**t for t in range(T)])
    oneN = np.transpose(np.matrix(np.ones(N)))
    betam = np.dot(oneN,betaT)
    costm = cB*qm+cR*am+hB*bom+hR*rom+pB*bsm+pR*rsm
    costdm = np.multiply(betam,costm)
    costdv = np.sum(costdm,axis=1)
    ec = np.sum(costdv)/N
    return ec,np.squeeze(np.asarray(costdv))

def cpsim_fbqsdr(d,L,T,s,b,r,Dv,Av,B,Zdict):
    Bv = np.zeros(T) # vector of total capacity
    sv = np.zeros(T) # vector of specimen
    bv = np.zeros(T) # vector of idle machine
    rv = np.zeros(T) # vector of raw material
    
    bsv = np.zeros(T) # vector of bioreactor shortage
    rsv = np.zeros(T) # vector of reagent shortage
    bov = np.zeros(T) # vector of bioreactor overstock
    rov = np.zeros(T) # vector of reagent overstock
    
    av = np.zeros(T) # vector of reagent replenishment
    qv = np.zeros(T) # vector of  bioreactor adjustment
    
    t = 0
    
    while t<T:
        Bv[t] = np.sum(b)
        sv[t] = s
        bv[t] = b[0]
        rv[t] = r
        
        a = get_a_iso(s,r,Zdict,Av[t],t,T)
        if t == 0:
            q = B
        else:
            q = 0 # B is fixed
        
        av[t] = a
        qv[t] = q
        
        m = np.min([s,b[0],r])
        s = s-m+Dv[t]
        b[0] = b[0]-m+q+b[1]
        for i in np.arange(1,L-1):
            b[i] = b[i+1]
        b[L-1] = m
        r = r-m+a
        
        bsv[t] = np.max([s-b[0],0])
        rsv[t] = np.max([s-r,0])
        bov[t] = np.max([b[0]-s,0])
        rov[t] = np.max([r-s,0])
        
        t = t+1
            
    return Bv, sv, bv, rv, bsv, rsv, bov, rov, av, qv

def solver_fbqsdr_p(d,s,b,r,Dtr,Dte,A,L,T,N,paras,p,beta):
    cR = paras[0]
    hR = paras[1]
    pR = paras[2]
    cB = paras[3]
    hB = paras[4]
    pB = paras[5]
    rhoB = (pB-(1-beta)*cB)/(pB+hB)
    q1 = max([0,poisson.ppf(rhoB,L*d)-b[0]])
    q2 = q1+1
    Bm1,am1,qm1,bsm1,rsm1,bom1,rom1,sm1,rm1 = test_fbqsdr_p(d,s,b,r,Dtr,A,L,T,N,paras,p,q1,beta)
    ec1,cv1 = cost_eval_iso(cB,cR,hB,hR,pB,pR,beta,am1,qm1,bom1,rom1,bsm1,rsm1)
    Bm2,am2,qm2,bsm2,rsm2,bom2,rom2,sm2,rm2 = test_fbqsdr_p(d,s,b,r,Dtr,A,L,T,N,paras,p,q2,beta)
    ec2,cv2 = cost_eval_iso(cB,cR,hB,hR,pB,pR,beta,am2,qm2,bom2,rom2,bsm2,rsm2)
    delta = ec2-ec1
    if delta==0: # output
        q = q1
    if delta<0: # search up
        while delta<0:
            ec1 = ec2
            q1 = q2
            q2 = q2+1
            Bm2,am2,qm2,bsm2,rsm2,bom2,rom2,sm2,rm2 = test_fbqsdr_p(d,s,b,r,Dtr,A,L,T,N,paras,p,q2,beta)
            ec2,cv2 = cost_eval_iso(cB,cR,hB,hR,pB,pR,beta,am2,qm2,bom2,rom2,bsm2,rsm2)
            delta = ec2-ec1
        q = q1
    if delta>0: # search down
        while delta>0:
            ec2 = ec1
            q2 = q1
            q1 = q1-1
            Bm1,am1,qm1,bsm1,rsm1,bom1,rom1,sm1,rm1 = test_fbqsdr_p(d,s,b,r,Dtr,A,L,T,N,paras,p,q1,beta)
            ec1,cv1 = cost_eval_iso(cB,cR,hB,hR,pB,pR,beta,am1,qm1,bom1,rom1,bsm1,rsm1)
            delta = ec2-ec1
            #print(delta)
        q = q2
    Bm,am,qm,bsm,rsm,bom,rom,sm,rm = test_fbqsdr_p(d,s,b,r,Dte,A,L,T,N,paras,p,q,beta)
    return Bm,am,qm,bsm,rsm,bom,rom,sm,rm

def test_fbqsdr_p(d,s,b,r,D,A,L,T,N,paras,p,q,beta):
    Bm = np.zeros((N,T))
    sm = np.zeros((N,T))
    bm = np.zeros((N,T))
    rm = np.zeros((N,T))
    
    bsm = np.zeros((N,T))
    rsm = np.zeros((N,T))
    bom = np.zeros((N,T))
    rom = np.zeros((N,T))
    
    am = np.zeros((N,T))
    qm = np.zeros((N,T))
    
    cR = paras[0]
    hR = paras[1]
    pR = paras[2]
    Zdict = get_Z_p(d, cR, hR, pR, beta, p)
        
    for n in range(N):
        Bm[n], sm[n], bm[n], rm[n], bsm[n], rsm[n], bom[n], rom[n], am[n], qm[n] = cpsim_fbqsdr(d,L,T,s,b,r,D[n],A[n],q,Zdict)

    return Bm,am,qm,bsm,rsm,bom,rom,sm,rm

# In[]: Co-Net functions
def cpsim_conet(d,L,T,s1,b1,r1,s2,b2,r2,Dv1,Dv2,Av1,Av2,B1,B2,method,paras,p,beta,cL=0):
    Bv1 = np.zeros(T)
    sv1 = np.zeros(T)
    bv1 = np.zeros(T)
    rv1 = np.zeros(T)
    
    bsv1 = np.zeros(T)
    rsv1 = np.zeros(T)
    bov1 = np.zeros(T)
    rov1 = np.zeros(T)
    
    av1 = np.zeros(T)
    qv1 = np.zeros(T)
    wv1 = np.zeros(T)
    ev1 = np.zeros(T)
    
    Bv2 = np.zeros(T)
    sv2 = np.zeros(T)
    bv2 = np.zeros(T)
    rv2 = np.zeros(T)
    
    bsv2 = np.zeros(T)
    rsv2 = np.zeros(T)
    bov2 = np.zeros(T)
    rov2 = np.zeros(T)
    
    av2 = np.zeros(T)
    qv2 = np.zeros(T)
    wv2 = np.zeros(T)
    ev2 = np.zeros(T)
    
    t = 0
    
    while t<T:
        Bv1[t] = np.sum(b1)
        sv1[t] = s1
        bv1[t] = b1[0]
        rv1[t] = r1
        
        Bv2[t] = np.sum(b2)
        sv2[t] = s2
        bv2[t] = b2[0]
        rv2[t] = r2
        
        if method == 'MYO':
            a1,a2,q1,q2,w1,w2,e1,e2 = get_action_myo(d,s1,b1,r1,Av1[t],s2,b2,r2,Av2[t],paras)
        if method == "MDL":
            a1,a2,q1,q2,w1,w2,e1,e2 = get_action_mdl(d,L,s1,b1,r1,Av1[t],s2,b2,r2,Av2[t],paras,cL,p,beta)
        # if method == "MRO":
        #     a1,a2,q1,q2,w1,w2,e1,e2 = get_action_mro()
        # if method == "IRO":
        #     a1,a2,q1,q2,w1,w2,e1,e2 = get_action_iro()
        
        if t == 0:
            q1 = q1+B1
            q2 = q2+B2
        
        av1[t] = a1
        qv1[t] = q1
        wv1[t] = w1
        ev1[t] = e1
        
        av2[t] = a2
        qv2[t] = q2
        wv2[t] = w2
        ev2[t] = e2
        
        # state transit
        m1 = np.min([s1,b1[0],r1])
        s1 = s1-m1+Dv1[t]+w1
        b1[0] = b1[0]-m1+q1+b1[1]
        for i in np.arange(1,L-1):
            b1[i] = b1[i+1]
        b1[L-1] = m1
        r1 = r1-m1+a1+e1
        
        m2 = np.min([s2,b2[0],r2])
        s2 = s2-m2+Dv2[t]+w2
        b2[0] = b2[0]-m2+q2+b2[1]
        for i in np.arange(1,L-1):
            b2[i] = b2[i+1]
        b2[L-1] = m2
        r2 = r2-m2+a2+e2        
        
        bsv1[t] = np.max([s1-b1[0],0])
        rsv1[t] = np.max([s1-r1,0])
        bov1[t] = np.max([b1[0]-s1,0])
        rov1[t] = np.max([r1-s1,0])
        
        bsv2[t] = np.max([s2-b2[0],0])
        rsv2[t] = np.max([s2-r2,0])
        bov2[t] = np.max([b2[0]-s2,0])
        rov2[t] = np.max([r2-s2,0])
        
        t = t+1    
    
    Keys = ['Bv', 'sv', 'bv', 'rv', 'bsv', 'rsv', 'bov', 'rov', 'av', 'qv', 'wv', 'ev']
    Sim1 = dict(zip(Keys,[Bv1, sv1, bv1, rv1, bsv1, rsv1, bov1, rov1, av1, qv1, wv1, ev1]))
    Sim2 = dict(zip(Keys,[Bv2, sv2, bv2, rv2, bsv2, rsv2, bov2, rov2, av2, qv2, wv2, ev2]))
                
    return Sim1,Sim2

def get_action_myo(d,s1,b1,r1,A1,s2,b2,r2,A2,paras):
    m1 = np.min([s1,b1[0],r1])
    m2 = np.min([s2,b2[0],r2])
    d_max = poisson.ppf(0.99,d)
    pdv = poisson.pmf(np.arange(d_max+1),d)
    cR = paras[0]
    hR = paras[1]
    pR = paras[2]
    # cB = paras[3]
    hB = paras[4]
    pB = paras[5]
    KS = paras[6]
    KR = paras[7]
    KB = paras[8]
    
    M_myo = gp.Model('get-action-myo')
    a = M_myo.addVars(2,obj=cR,lb=0,vtype=GRB.INTEGER,name='a')
    q = M_myo.addVars(2,obj=0,vtype=GRB.INTEGER,name='q')
    qq = M_myo.addVars(2,obj=KB/2,vtype=GRB.CONTINUOUS,name='qq')
    w = M_myo.addVars(2,obj=0,vtype=GRB.INTEGER,name='w')
    ww = M_myo.addVars(2,obj=KS/2,vtype=GRB.CONTINUOUS,name='ww')
    e = M_myo.addVars(2,obj=0,vtype=GRB.INTEGER,name='e')
    ee = M_myo.addVars(2,obj=KR/2,vtype=GRB.CONTINUOUS,name='ee')
    Ord = M_myo.addVars(2,d_max+1,obj=hR*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Ord')
    Srd = M_myo.addVars(2,d_max+1,obj=pR*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Srd')
    Obd = M_myo.addVars(2,d_max+1,obj=hB*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Obd')
    Sbd = M_myo.addVars(2,d_max+1,obj=pB*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Sbd')
    
    if A1 == 0:
        M_myo.addConstr(a[0]==0)
    if A2 == 0:
        M_myo.addConstr(a[1]==0)
    
    M_myo.addConstrs((qq[i]-q[i]>=0 for i in range(2)))
    M_myo.addConstrs((qq[i]+q[i]>=0 for i in range(2)))
    M_myo.addConstrs((ww[i]-w[i]>=0 for i in range(2)))
    M_myo.addConstrs((ww[i]+w[i]>=0 for i in range(2)))
    M_myo.addConstrs((ee[i]-e[i]>=0 for i in range(2)))
    M_myo.addConstrs((ee[i]+e[i]>=0 for i in range(2)))
    
    M_myo.addConstr(q[0]+q[1]==0)
    M_myo.addConstr(w[0]+w[1]==0)
    M_myo.addConstr(e[0]+e[1]==0)
    M_myo.addConstr(w[0]+s1-m1>=0)
    M_myo.addConstr(w[1]+s2-m2>=0)
    M_myo.addConstr(e[0]+a[0]+r1-m1>=0)
    M_myo.addConstr(e[1]+a[1]+r2-m2>=0)
    M_myo.addConstr(q[0]+b1[0]+b1[1]-m1>=0)
    M_myo.addConstr(q[1]+b2[0]+b2[1]-m2>=0)
    
    M_myo.addConstrs((Ord[0,i]-r1-a[0]-e[0]+s1+i+w[0]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Ord[1,i]-r2-a[1]-e[1]+s2+i+w[1]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Srd[0,i]+r1+a[0]+e[0]-s1-i-w[0]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Srd[1,i]+r2+a[1]+e[1]-s2-i-w[i]>=0 for i in range(d_max+1)))
    
    M_myo.addConstrs((Obd[0,i]-b1[0]-b1[1]-q[0]+s1+i+w[0]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Obd[1,i]-b2[0]-b2[1]-q[1]+s2+i+w[1]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Sbd[0,i]+b1[0]+b1[1]+q[0]-s1-i-w[0]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Sbd[1,i]+b2[0]+b2[1]+q[1]-s2-i-w[i]>=0 for i in range(d_max+1)))
    
    M_myo.setParam('OutputFlag',False)
    M_myo.optimize()
    
    a1 = a[0].X
    a2 = a[1].X
    q1 = q[0].X
    q2 = q[1].X
    w1 = w[0].X
    w2 = w[1].X
    e1 = e[0].X
    e2 = e[1].X
    
    return a1,a2,q1,q2,w1,w2,e1,e2

def get_action_mdl(d,L,s1,b1,r1,A1,s2,b2,r2,A2,paras,cL,p,beta):
    m1 = np.min([s1,b1[0],r1])
    m2 = np.min([s2,b2[0],r2])
    d_max = poisson.ppf(0.99,d)
    pdv = poisson.pmf(np.arange(d_max+1),d)
    cR = paras[0]
    hR = paras[1]
    pR = paras[2]
    # cB = paras[3]
    hB = paras[4]
    pB = paras[5]
    KS = paras[6]
    KR = paras[7]
    KB = paras[8]
    
    M_mdl = gp.Model('get-action-mdl')
    
    # period 0:
    a0 = M_mdl.addVars(2,obj=cR,lb=0,vtype=GRB.INTEGER,name='a0')
    q0 = M_mdl.addVars(2,obj=0,vtype=GRB.INTEGER,name='q0')
    qq0 = M_mdl.addVars(2,obj=KB/2,vtype=GRB.CONTINUOUS,name='qq0')
    w0 = M_mdl.addVars(2,obj=0,vtype=GRB.INTEGER,name='w0')
    ww0 = M_mdl.addVars(2,obj=KS/2,vtype=GRB.CONTINUOUS,name='ww0')
    e0 = M_mdl.addVars(2,obj=0,vtype=GRB.INTEGER,name='e0')
    ee0 = M_mdl.addVars(2,obj=KR/2,vtype=GRB.CONTINUOUS,name='ee0')
    Ord0 = M_mdl.addVars(2,d_max+1,obj=hR*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Ord0')
    Srd0 = M_mdl.addVars(2,d_max+1,obj=pR*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Srd0')
    Obd0 = M_mdl.addVars(2,d_max+1,obj=hB*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Obd0')
    Sbd0 = M_mdl.addVars(2,d_max+1,obj=pB*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Sbd0')
    
    if A1 == 1:
        M_mdl.addConstr(a0[0]==0)
    if A2 == 1:
        M_mdl.addConstr(a0[1]==0)
    
    M_mdl.addConstrs((qq0[i]-q0[i]>=0 for i in range(2)))
    M_mdl.addConstrs((qq0[i]+q0[i]>=0 for i in range(2)))
    M_mdl.addConstrs((ww0[i]-w0[i]>=0 for i in range(2)))
    M_mdl.addConstrs((ww0[i]+w0[i]>=0 for i in range(2)))
    M_mdl.addConstrs((ee0[i]-e0[i]>=0 for i in range(2)))
    M_mdl.addConstrs((ee0[i]+e0[i]>=0 for i in range(2)))
    
    M_mdl.addConstr(q0[0]+q0[1]==0)
    M_mdl.addConstr(w0[0]+w0[1]==0)
    M_mdl.addConstr(e0[0]+e0[1]==0)
    M_mdl.addConstr(w0[0]+s1-m1>=0)
    M_mdl.addConstr(w0[1]+s2-m2>=0)
    M_mdl.addConstr(e0[0]+a0[0]+r1-m1>=0)
    M_mdl.addConstr(e0[1]+a0[1]+r2-m2>=0)
    M_mdl.addConstr(q0[0]+b1[0]+b1[1]-m1>=0)
    M_mdl.addConstr(q0[1]+b2[0]+b2[1]-m2>=0)
    
    M_mdl.addConstrs((Ord0[0,i]-r1-a0[0]-e0[0]+s1+i+w0[0]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Ord0[1,i]-r2-a0[1]-e0[1]+s2+i+w0[1]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Srd0[0,i]+r1+a0[0]+e0[0]-s1-i-w0[0]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Srd0[1,i]+r2+a0[1]+e0[1]-s2-i-w0[i]>=0 for i in range(d_max+1)))
    
    M_mdl.addConstrs((Obd0[0,i]-b1[0]-b1[1]-q0[0]+s1+i+w0[0]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Obd0[1,i]-b2[0]-b2[1]-q0[1]+s2+i+w0[1]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Sbd0[0,i]+b1[0]+b1[1]+q0[0]-s1-i-w0[0]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Sbd0[1,i]+b2[0]+b2[1]+q0[1]-s2-i-w0[i]>=0 for i in range(d_max+1)))
    
    # period 1:
    if cL>0:
        pAm = np.array([[1-p,p],[1-p,p]])
        a1 = M_mdl.addVars(2,2,obj=beta*cR*pAm,lb=0,vtype=GRB.INTEGER,name='a1')
        q1 = M_mdl.addVars(2,2,obj=0,vtype=GRB.INTEGER,name='q1')
        qq1 = M_mdl.addVars(2,2,obj=beta*KB/2*pAm,vtype=GRB.CONTINUOUS,name='qq1')
        w1 = M_mdl.addVars(2,2,obj=0,vtype=GRB.INTEGER,name='w1')
        ww1 = M_mdl.addVars(2,2,obj=beta*KS/2*pAm,vtype=GRB.CONTINUOUS,name='ww1')
        e1 = M_mdl.addVars(2,2,obj=0,vtype=GRB.INTEGER,name='e1')
        ee1 = M_mdl.addVars(2,2,obj=beta*KR/2*pAm,vtype=GRB.CONTINUOUS,name='ee1')
        pAdv1 = (1-p)*pdv
        pAdv2 = p*pdv
        pAdm = np.array([pAdv1,pAdv2])
        pAdt = np.array([pAdm,pAdm])
        Ord1 = M_mdl.addVars(2,2,d_max+1,obj=beta*hR*pAdt,lb=0,vtype=GRB.CONTINUOUS,name='Ord1')
        Srd1 = M_mdl.addVars(2,2,d_max+1,obj=beta*pR*pAdt,lb=0,vtype=GRB.CONTINUOUS,name='Srd1')
        Obd1 = M_mdl.addVars(2,2,d_max+1,obj=beta*hB*pAdt,lb=0,vtype=GRB.CONTINUOUS,name='Obd1')
        Sbd1 = M_mdl.addVars(2,2,d_max+1,obj=beta*pB*pAdt,lb=0,vtype=GRB.CONTINUOUS,name='Sbd1')
        
        M_mdl.addConstrs((a1[i,1]==0 for i in range(2)))
        M_mdl.addConstrs((qq1[i,j]-q1[i,j]>=0 for i in range(2) for j in range(2)))
        M_mdl.addConstrs((qq1[i,j]+q1[i,j]>=0 for i in range(2) for j in range(2)))
        M_mdl.addConstrs((ww1[i,j]-w1[i,j]>=0 for i in range(2) for j in range(2)))
        M_mdl.addConstrs((ww1[i,j]+w1[i,j]>=0 for i in range(2) for j in range(2)))
        M_mdl.addConstrs((ee1[i,j]-e1[i,j]>=0 for i in range(2) for j in range(2)))
        M_mdl.addConstrs((ee1[i,j]+e1[i,j]>=0 for i in range(2) for j in range(2)))
        
        M_mdl.addConstrs((q1[0,j]+q1[1,j]==0 for j in range(2)))
        M_mdl.addConstrs((w1[0,j]+w1[1,j]==0 for j in range(2)))
        M_mdl.addConstrs((e1[0,j]+e1[1,j]==0 for j in range(2)))
        
        wl1 = M_mdl.addVars(2,obj=0,vtype=GRB.CONTINUOUS,name='wl1')
        M_mdl.addConstrs((w1[0,j]-wl1[0]>=0 for j in range(2)))
        M_mdl.addConstrs((w1[1,j]-wl1[1]>=0 for j in range(2)))
        M_mdl.addGenConstrMin(wl1[0],[0,b1[0]+b1[1]+q0[0]-s1-d-w0[0],r1+a0[0]+e0[0]-s1-d-w0[0]])
        M_mdl.addGenConstrMin(wl1[1],[0,b2[0]+b2[1]+q0[1]-s2-d-w0[1],r2+a0[1]+e0[1]-s2-d-w0[1]])
        
        el1 = M_mdl.addVars(2,obj=0,vtype=GRB.CONTINUOUS,name='el1')
        M_mdl.addConstrs((e1[0,j]+a1[0,j]-el1[0]>=0 for j in range(2)))
        M_mdl.addConstrs((e1[1,j]+a1[1,j]-el1[1]>=0 for j in range(2)))
        M_mdl.addGenConstrMin(el1[0],[0,b1[0]+b1[1]+q0[0]-r1-a0[0]-e0[0],s1+d+w0[0]-r1-a0[0]-e0[0]])
        M_mdl.addGenConstrMin(el1[1],[0,b2[0]+b2[1]+q0[1]-r2-a0[1]-e0[1],s2+d+w0[1]-r2-a0[1]-e0[1]])
        
        ql1 = M_mdl.addVars(2,obj=0,vtype=GRB.CONTINUOUS,name='ql1')
        M_mdl.addConstrs((q1[0,j]+b1[2]-ql1[0]>=0 for j in range(2)))
        M_mdl.addConstrs((q1[1,j]+b1[2]-ql1[1]>=0 for j in range(2)))
        M_mdl.addGenConstrMin(ql1[0],[0,s1+d+w0[0]-b1[0]-b1[1]-q0[0],r1+a0[0]+e0[0]-b1[0]-b1[1]-q0[0]])
        M_mdl.addGenConstrMin(ql1[1],[0,s2+d+w0[1]-b2[0]-b2[1]-q0[1],r2+a0[1]+e0[1]-b2[0]-b2[1]-q0[1]])
        
        M_mdl.addConstrs((Ord1[0,j,k]-r1-a0[0]-e0[0]-a1[0,j]-e1[0,j]+s1+d+w0[0]+k+w1[0,j]>=0 for j in range(2) for k in range(d_max+1)))
        M_mdl.addConstrs((Ord1[1,j,k]-r2-a0[1]-e0[1]-a1[1,j]-e1[1,j]+s2+d+w0[1]+k+w1[1,j]>=0 for j in range(2) for k in range(d_max+1)))
        M_mdl.addConstrs((Srd1[0,j,k]+r1+a0[0]+e0[0]+a1[0,j]+e1[0,j]-s1-d-w0[0]-k-w1[0,j]>=0 for j in range(2) for k in range(d_max+1)))
        M_mdl.addConstrs((Srd1[1,j,k]+r2+a0[1]+e0[1]+a1[1,j]+e1[1,j]-s2-d-w0[1]-k-w1[1,j]>=0 for j in range(2) for k in range(d_max+1)))
        
        M_mdl.addConstrs((Obd1[0,j,k]-b1[0]-b1[1]-q0[0]-q1[0,j]-b1[2]+s1+d+w0[0]+k+w1[0,j]>=0 for j in range(2) for k in range(d_max+1)))
        M_mdl.addConstrs((Obd1[1,j,k]-b2[0]-b2[1]-q0[1]-q1[1,j]-b2[2]+s2+d+w0[1]+k+w1[1,j]>=0 for j in range(2) for k in range(d_max+1)))
        M_mdl.addConstrs((Sbd1[0,j,k]+b1[0]+b1[1]+q0[0]+q1[0,j]+b1[2]-s1-d-w0[0]-k-w1[0,j]>=0 for j in range(2) for k in range(d_max+1)))
        M_mdl.addConstrs((Sbd1[1,j,k]+b2[0]+b2[1]+q0[1]+q1[1,j]+b2[2]-s2-d-w0[1]-k-w1[1,j]>=0 for j in range(2) for k in range(d_max+1)))

    # period 2
    if cL>1:
        pAAm = np.array([[(1-p)**2,(1-p)*p,p*(1-p),p**2],[(1-p)**2,(1-p)*p,p*(1-p),p**2]])
        a2 = M_mdl.addVars(2,4,obj=beta**2*cR*pAAm,vtype=GRB.INTEGER,name='a2')
        q2 = M_mdl.addVars(2,4,obj=0,vtype=GRB.INTEGER,name='q2')
        qq2 = M_mdl.addVars(2,4,obj=beta**2*KB/2*pAAm,vtype=GRB.CONTINUOUS,name='qq2')
        w2 = M_mdl.addVars(2,4,obj=0,vtype=GRB.INTEGER,name='w2')
        ww2 = M_mdl.addVars(2,4,obj=beta**2*KS/2*pAAm,vtype=GRB.CONTINUOUS,name='ww2')
        e2 = M_mdl.addVars(2,4,obj=0,vtype=GRB.INTEGER,name='e2')
        ee2 = M_mdl.addVars(2,4,obj=beta**2*KR/2*pAAm,vtype=GRB.CONTINUOUS,name='ee2')
        pAAdm = np.array([(1-p)**2*pdv,(1-p)*p*pdv,p*(1-p)*pdv,p**2*pdv])
        pAAdt = np.array([pAAdm,pAAdm])
        Ord2 = M_mdl.addVars(2,4,d_max+1,obj=beta**2*hR*pAAdt,lb=0,vtype=GRB.CONTINUOUS,name='Ord2')
        Srd2 = M_mdl.addVars(2,4,d_max+1,obj=beta**2*pR*pAAdt,lb=0,vtype=GRB.CONTINUOUS,name='Srd2')
        Obd2 = M_mdl.addVars(2,4,d_max+1,obj=beta**2*hB*pAAdt,lb=0,vtype=GRB.CONTINUOUS,name='Obd2')
        Sbd2 = M_mdl.addVars(2,4,d_max+1,obj=beta**2*pB*pAAdt,lb=0,vtype=GRB.CONTINUOUS,name='Sbd2')
        
        M_mdl.addConstrs((a2[i,j]==0 for i in range(2) for j in [1,3]))
        M_mdl.addConstrs((qq2[i,j]-q2[i,j]>=0 for i in range(2) for j in range(4)))
        M_mdl.addConstrs((qq2[i,j]+q2[i,j]>=0 for i in range(2) for j in range(4)))
        M_mdl.addConstrs((ww2[i,j]-w2[i,j]>=0 for i in range(2) for j in range(4)))
        M_mdl.addConstrs((ww2[i,j]+w2[i,j]>=0 for i in range(2) for j in range(4)))
        M_mdl.addConstrs((ee2[i,j]-e2[i,j]>=0 for i in range(2) for j in range(4)))
        M_mdl.addConstrs((ee2[i,j]+e2[i,j]>=0 for i in range(2) for j in range(4)))
        
        M_mdl.addConstrs((q2[0,j]+q2[1,j]==0 for j in range(4)))
        M_mdl.addConstrs((w2[0,j]+w2[1,j]==0 for j in range(4)))
        M_mdl.addConstrs((e2[0,j]+e2[1,j]==0 for j in range(4)))
        
        wl2 = M_mdl.addVars(2,2,obj=0,vtype=GRB.CONTINUOUS,name='wl2')
        M_mdl.addConstrs((w2[0,j]-wl2[0,0]>=0 for j in [0,1]))
        M_mdl.addConstrs((w2[0,j]-wl2[0,1]>=0 for j in [2,3]))
        M_mdl.addConstrs((w2[1,j]-wl2[1,0]>=0 for j in [0,1]))
        M_mdl.addConstrs((w2[1,j]-wl2[1,1]>=0 for j in [2,3]))
        M_mdl.addGenConstrMin(wl2[0,0],[0,b1[0]+b1[1]+q0[0]+q1[0,0]-s1-2*d-w0[0]-w1[0,0],r1+a0[0]+e0[0]+a1[0,0]+e1[0,0]-s1-2*d-w0[0]-w1[0,0]])
        M_mdl.addGenConstrMin(wl2[0,1],[0,b1[0]+b1[1]+q0[0]+q1[0,1]-s1-2*d-w0[0]-w1[0,1],r1+a0[0]+e0[0]+a1[0,1]+e1[0,1]-s1-2*d-w0[0]-w1[0,1]])
        M_mdl.addGenConstrMin(wl2[1,0],[0,b2[0]+b2[1]+q0[1]+q1[1,0]-s1-2*d-w0[1]-w1[1,0],r2+a0[1]+e0[1]+a1[1,0]+e1[1,0]-s2-2*d-w0[1]-w1[1,0]])
        M_mdl.addGenConstrMin(wl2[1,1],[0,b2[0]+b2[1]+q0[1]+q1[1,1]-s1-2*d-w0[1]-w1[1,1],r2+a0[1]+e0[1]+a1[1,1]+e1[1,1]-s2-2*d-w0[1]-w1[1,1]])
        
        el2 = M_mdl.addVars(2,2,obj=0,vtype=GRB.CONTINUOUS,name='el2')
        M_mdl.addConstrs((e2[0,j]+a2[0,j]-el2[0,0]>=0 for j in [0,1]))
        M_mdl.addConstrs((e2[0,j]+a2[0,j]-el2[0,1]>=0 for j in [2,3]))
        M_mdl.addConstrs((e2[1,j]+a2[1,j]-el2[1,0]>=0 for j in [0,1]))
        M_mdl.addConstrs((e2[1,j]+a2[1,j]-el2[1,1]>=0 for j in [2,3]))
        M_mdl.addGenConstrMin(el2[0,0],[0,s1+2*d+w0[0]+w1[0,0]-r1-a0[0]-e0[0]-a1[0,0]-e1[0,0],b1[0]+b1[1]+q0[0]+q1[0,0]-r1-a0[0]-e0[0]-a1[0,0]-e1[0,0]])
        M_mdl.addGenConstrMin(el2[0,1],[0,s1+2*d+w0[0]+w1[0,1]-r1-a0[0]-e0[0]-a1[0,1]-e1[0,1],b1[0]+b1[1]+q0[0]+q1[0,1]-r1-a0[0]-e0[0]-a1[0,1]-e1[0,1]])
        M_mdl.addGenConstrMin(el2[1,0],[0,s2+2*d+w0[1]+w1[1,0]-r2-a0[1]-e0[1]-a1[1,0]-e1[1,0],b2[0]+b2[1]+q0[1]+q1[1,0]-r2-a0[1]-e0[1]-a1[1,0]-e1[1,0]])
        M_mdl.addGenConstrMin(el2[1,1],[0,s2+2*d+w0[1]+w1[1,1]-r2-a0[1]-e0[1]-a1[1,1]-e1[1,1],b2[0]+b2[1]+q0[1]+q1[1,1]-r2-a0[1]-e0[1]-a1[1,1]-e1[1,1]])
        
        ql2 = M_mdl.addVars(2,2,obj=0,vtype=GRB.CONTINUOUS,name='ql2')
        M_mdl.addConstrs((q2[0,j]+m1-ql2[0,0]>=0 for j in [0,1]))
        M_mdl.addConstrs((q2[0,j]+m1-ql2[0,1]>=0 for j in [2,3]))
        M_mdl.addConstrs((q2[1,j]+m2-ql2[1,0]>=0 for j in [0,1]))
        M_mdl.addConstrs((q2[1,j]+m2-ql2[1,1]>=0 for j in [2,3]))
        M_mdl.addGenConstrMin(ql2[0,0],[0,s1+2*d+w0[0]+w1[0,0]-b1[0]-b1[1]-q0[0]-q1[0,0],r1+a0[0]+e0[0]+a1[0,0]+e1[0,0]-b1[0]-b1[1]-q0[0]-q1[0,0]])
        M_mdl.addGenConstrMin(ql2[0,1],[0,s1+2*d+w0[0]+w1[0,1]-b1[0]-b1[1]-q0[0]-q1[0,1],r1+a0[0]+e0[0]+a1[0,1]+e1[0,1]-b1[0]-b1[1]-q0[0]-q1[0,1]])
        M_mdl.addGenConstrMin(ql2[1,0],[0,s2+2*d+w0[1]+w1[1,0]-b2[0]-b2[1]-q0[1]-q1[1,0],r2+a0[1]+e0[1]+a1[1,0]+e1[1,0]-b2[0]-b2[1]-q0[1]-q1[1,0]])
        M_mdl.addGenConstrMin(ql2[1,1],[0,s2+2*d+w0[1]+w1[1,1]-b2[0]-b2[1]-q0[1]-q1[1,1],r2+a0[1]+e0[1]+a1[1,1]+e1[1,1]-b2[0]-b2[1]-q0[1]-q1[1,1]])
        
        M_mdl.addConstrs((Ord2[0,j,k]-r1-a0[0]-e0[0]-a1[0,0]-e1[0,0]-a2[0,j]-e2[0,j]+s1+2*d+w0[0]+w1[0,0]+k+w2[0,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Ord2[0,j,k]-r1-a0[0]-e0[0]-a1[0,1]-e1[0,1]-a2[0,j]-e2[0,j]+s1+2*d+w0[0]+w1[0,1]+k+w2[0,j]>=0 for j in [2,3] for k in range(d_max+1)))
        M_mdl.addConstrs((Ord2[1,j,k]-r2-a0[1]-e0[1]-a1[1,0]-e1[1,0]-a2[1,j]-e2[1,j]+s2+2*d+w0[1]+w1[1,0]+k+w2[1,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Ord2[1,j,k]-r2-a0[1]-e0[1]-a1[1,1]-e1[1,1]-a2[1,j]-e2[1,j]+s2+2*d+w0[1]+w1[1,1]+k+w2[1,j]>=0 for j in [2,3] for k in range(d_max+1)))
        
        M_mdl.addConstrs((Srd2[0,j,k]+r1+a0[0]+e0[0]+a1[0,0]+e1[0,0]+a2[0,j]+e2[0,j]-s1-2*d-w0[0]-w1[0,0]-k-w2[0,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Srd2[0,j,k]+r1+a0[0]+e0[0]+a1[0,1]+e1[0,1]+a2[0,j]+e2[0,j]-s1-2*d-w0[0]-w1[0,1]-k-w2[0,j]>=0 for j in [2,3] for k in range(d_max+1)))
        M_mdl.addConstrs((Srd2[1,j,k]+r2+a0[1]+e0[1]+a1[1,0]+e1[1,0]+a2[1,j]+e2[1,j]-s2-2*d-w0[1]-w1[1,0]-k-w2[1,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Srd2[1,j,k]+r2+a0[1]+e0[1]+a1[1,1]+e1[1,1]+a2[1,j]+e2[1,j]-s2-2*d-w0[1]-w1[1,1]-k-w2[1,j]>=0 for j in [2,3] for k in range(d_max+1)))
        
        M_mdl.addConstrs((Obd2[0,j,k]-b1[0]-b1[1]-q0[0]-q1[0,0]-b1[2]-q2[0,j]-m1+s1+2*d+w0[0]+w1[0,0]+k+w2[0,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Obd2[0,j,k]-b1[0]-b1[1]-q0[0]-q1[0,1]-b1[2]-q2[0,j]-m1+s1+2*d+w0[0]+w1[0,1]+k+w2[0,j]>=0 for j in [2,3] for k in range(d_max+1)))
        M_mdl.addConstrs((Obd2[1,j,k]-b1[0]-b1[1]-q0[1]-q1[1,0]-b1[2]-q2[1,j]-m2+s2+2*d+w0[1]+w1[1,0]+k+w2[1,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Obd2[1,j,k]-b1[0]-b1[1]-q0[1]-q1[1,1]-b1[2]-q2[1,j]-m2+s2+2*d+w0[1]+w1[1,1]+k+w2[1,j]>=0 for j in [2,3] for k in range(d_max+1)))
        
        M_mdl.addConstrs((Sbd2[0,j,k]+b1[0]+b1[1]+q0[0]+q1[0,0]+b1[2]+q2[0,j]+m1-s1-2*d-w0[0]-w1[0,0]-k-w2[0,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Sbd2[0,j,k]+b1[0]+b1[1]+q0[0]+q1[0,1]+b1[2]+q2[0,j]+m1-s1-2*d-w0[0]-w1[0,1]-k-w2[0,j]>=0 for j in [2,3] for k in range(d_max+1)))
        M_mdl.addConstrs((Sbd2[1,j,k]+b1[0]+b1[1]+q0[1]+q1[1,0]+b1[2]+q2[1,j]+m2-s2-2*d-w0[1]-w1[1,0]-k-w2[1,j]>=0 for j in [0,1] for k in range(d_max+1)))
        M_mdl.addConstrs((Sbd2[1,j,k]+b1[0]+b1[1]+q0[1]+q1[1,1]+b1[2]+q2[1,j]+m2-s2-2*d-w0[1]-w1[1,1]-k-w2[1,j]>=0 for j in [2,3] for k in range(d_max+1)))

    # optimize
    M_mdl.setParam('OutputFlag',False)
    M_mdl.optimize()
    
    # output
    a1o = a0[0].X
    a2o = a0[1].X
    q1o = q0[0].X
    q2o = q0[1].X
    w1o = w0[0].X
    w2o = w0[1].X
    e1o = e0[0].X
    e2o = e0[1].X
    
    return a1o,a2o,q1o,q2o,w1o,w2o,e1o,e2o