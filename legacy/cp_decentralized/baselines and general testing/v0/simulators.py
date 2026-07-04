#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:49:04 2020

@author: junxuanli

System simulators and data collector
"""

import numpy as np
from scipy.stats import poisson
from scipy.stats import bernoulli
import gurobipy as gp
from gurobipy import GRB

# In[]: Stochasticity generator
def d_sample(d,T,N): # Random demand sampler
    # d: Poisson demand rate
    # T: Planning horizon
    # N: Number of simulation runs
    D = poisson.rvs(d,size=(N,T),random_state=0)
    return D

def A_sample_cor(p,T,N,cor): # Random supplier state generator
    # p: prob. of disruption (supplier state: 0 - undisrupted, 1 - disrupted)
    # T: Planning horizon
    # N: Number of simulation runs
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

# In[]: Iso-Net simulator
def get_Z_p(d,cR,hR,pR,beta,p): # Calculate base-stock level look-up table
    # d: Poisson demand rate
    # cR: Reagent purchase cost
    # hR: Reagent holding cost
    # pR: Reagent penalty cost
    # beta: Discount factor
    # p: Disruption probability
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

def get_a_iso(s,r,Zdict,A,t,T): # Calculate reagent replenishment quantity
    # s: Specimen queue length
    # r: On-hand inventory
    # Zdict: Iso-Net look-up table
    # A: Supplier state
    # t: Current epoch
    # T: Planning horizon
    if A==1:
        a = 0
    else:
        H = min([29,T-t])    
        y = Zdict[H]+s        
        a = max([0,y-r])
    return a

def cpsim_isonet_single(d,L,T,s,b,r,Dv,Av,B,Zdict): # Iso-Net single facility simulation (need to run this for each facility)
    # d: Poisson demand rate
    # L: Production leadtime
    # T: Planning horizon
    # N: Number of simulation runs
    # s: Specimen queue length
    # b: Bioreactor vector
    # r: On-hand inventory
    # Dv: Simulated demand scenario
    # Av: Simulated supplier state scenario
    # B: Simulated idle bioreactor adjustment
    # Zdict: Iso-Net look-up table
    Bv = np.zeros(T) # Total number of bioreactor v.s. time
    sv = np.zeros(T) # Number of specimens v.s. time
    bv = np.zeros(T) # Number of idle machine v.s. time
    rv = np.zeros(T) # Number of reagent v.s. time
    
    bsv = np.zeros(T) # Idle bioreactor shortage v.s. time
    rsv = np.zeros(T) # Reagent shortage v.s. time
    bov = np.zeros(T) # Idle bioreactor overstock v.s. time
    rov = np.zeros(T) # Reagent overstock v.s. time
    
    av = np.zeros(T) # Reagent replenishment v.s. time
    qv = np.zeros(T) # Idle bioreactor adjustment v.s. time
    
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
            
    return Bv, sv, bv, rv, bsv, rsv, bov, rov, av, qv # output system states, decisions, overstocks and understocks

def cost_eval_iso(cB,cR,hB,hR,pB,pR,beta,am,qm,bom,rom,bsm,rsm): # Calculate total costs (need to run this for each facility)
    # cB: Bioreactor purchase cost
    # cR: Reagent purchase cost
    # hB: Idle bioreactor holding cost
    # hR: Reagent holding cost
    # pB: Idle bioreactor penatly cost
    # pR: Reagent penalty cost
    # beta: Discount factor
    # am: Replenishment quantity matrix
    # qm: Idle bioreactor adjustment matrix
    # bom: Bioreactor overstock matrix
    # rom: Reagent overstock matrix
    # bsm: Bioreactor shortage matrix
    # rsm: Reagent shortage matrix
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

# In[]: MYO Co-Net simulator
def cpsim_conet(d,L,T,s1,b1,r1,s2,b2,r2,Dv1,Dv2,Av1,Av2,B1,B2,paras):
    # d: Poisson demand rate
    # L: Production leadtime
    # T: Planning horizon
    # N: Number of simulation runs
    # s: Specimen queue length
    # b: Bioreactor vector
    # r: On-hand inventory
    # D: Simulated demand realization
    # A: Simulated supplier state realization
    # B: Simulated idle bioreactor adjustment
    # paras: Cost parameters
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
        
        a1,a2,q1,q2,w1,w2,e1,e2 = get_action_myo(d,s1,b1,r1,Av1[t],s2,b2,r2,Av2[t],paras)
        
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


# In[]: Single phase general action simulator
def cpsim_conet_onephase(action,L,s1,b1,r1,s2,b2,r2,D1,D2,A1,A2,B1,B2,paras):
    # action: actions - a1,a2,q1,q2,w1,w2,e1,e2
    # L: Production leadtime
    # N: Number of simulation runs
    # s: Specimen queue length
    # b: Bioreactor vector
    # r: On-hand inventory
    # D: Simulated demand realization
    # A: Simulated supplier state realization
    # B: Simulated idle bioreactor adjustment
    a1 = action[0]
    a2 = action[1]
    q1 = action[2]
    q2 = action[3]
    w1 = action[4]
    w2 = action[5]
    e1 = action[6]
    e2 = action[7]
    
    # state transit
    m1 = np.min([s1,b1[0],r1])
    s1 = s1-m1+D1+w1
    b1[0] = b1[0]-m1+q1+b1[1]+B1
    for i in np.arange(1,L-1):
        b1[i] = b1[i+1]
    b1[L-1] = m1
    r1 = r1-m1+a1+e1
        
    m2 = np.min([s2,b2[0],r2])
    s2 = s2-m2+D2+w2
    b2[0] = b2[0]-m2+q2+b2[1]+B2
    for i in np.arange(1,L-1):
        b2[i] = b2[i+1]
    b2[L-1] = m2
    r2 = r2-m2+a2+e2
    
    bs1 = np.max([s1-b1[0],0])
    rs1 = np.max([s1-r1,0])
    bo1 = np.max([b1[0]-s1,0])
    ro1 = np.max([r1-s1,0])
        
    bs2 = np.max([s2-b2[0],0])
    rs2 = np.max([s2-r2,0])
    bo2 = np.max([b2[0]-s2,0])
    ro2 = np.max([r2-s2,0])
    
    cR = paras[0]
    hR = paras[1]
    pR = paras[2]
    cB = paras[3]
    hB = paras[4]
    pB = paras[5]
    KS = paras[6]
    KR = paras[7]
    KB = paras[8]
    
    cost = cR*(a1+a2)+cB*(B1+B2)+hR*(ro1+ro2)+hB*(bo1+bo2)+pR*(rs1+rs2)+pB*(bs1+bs2)+KS*abs(w1)+KR*abs(e1)+KB*abs(q1)
              
    return s1,b1,r1,s2,b2,r2,cost


# In[]: Function approximation data collector
