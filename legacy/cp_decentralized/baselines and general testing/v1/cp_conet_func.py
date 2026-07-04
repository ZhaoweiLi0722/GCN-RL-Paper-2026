#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 00:14:29 2020

@author: junxuanli

Co-Net functions -- two facilities network
"""

import numpy as np
from scipy.stats import poisson
import gurobipy as gp
from gurobipy import GRB
from cp_isonet_func import get_Zdict
import copy

# In[]:
def sim_co(d,L,T,x,Dv,Av,B,method,kwargs):
    # kwargs: paras, p, beta, cL, Zdict
    sim_res = {}
    
    Bv1 = np.zeros(T)
    sv1 = np.zeros(T)
    b0v1 = np.zeros(T)
    rv1 = np.zeros(T)
        
    av1 = np.zeros(T)
    qv1 = np.zeros(T)
    wv1 = np.zeros(T)
    ev1 = np.zeros(T)
    
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
    wv2 = np.zeros(T)
    ev2 = np.zeros(T)
    
    bsv2 = np.zeros(T)
    rsv2 = np.zeros(T)
    bov2 = np.zeros(T)
    rov2 = np.zeros(T)
    
    t = 0
    while t < T:
        Bv1[t] = np.sum(x[0]['b'])
        Bv2[t] = np.sum(x[1]['b'])
        sv1[t] = x[0]['s']
        sv2[t] = x[1]['s']
        b0v1[t] = x[0]['b'][0]
        b0v2[t] = x[1]['b'][0]
        rv1[t] = x[0]['r']
        rv2[t] = x[1]['r']
        
        if method == 'myo':
            act = get_act_myo(d,x,[Av[0][t],Av[1][t]],kwargs['paras'])
        if method == 'emyo':
            act = get_act_emyo(d,x,[Av[0][t],Av[1][t]],t,T,kwargs['paras'],kwargs['Zdict'])
        if method == 'mdl1':
            act = get_act_mdl(d,x,[Av[0][t],Av[1][t]],t,T,kwargs['paras'],min([T-t-1,1]),kwargs['p'],kwargs['beta'])
        if method == 'mdl2':
            act = get_act_mdl(d,x,[Av[0][t],Av[1][t]],t,T,kwargs['paras'],min([T-t-1,2]),kwargs['p'],kwargs['beta'])
        
        av1[t], qv1[t], wv1[t], ev1[t] = act[0]['a'], act[0]['q'], act[0]['w'], act[0]['e']
        av2[t], qv2[t], wv2[t], ev2[t] = act[1]['a'], act[1]['q'], act[1]['w'], act[1]['e']
        
        m1 = np.min([x[0]['s'],x[0]['b'][0],x[0]['r']])
        m2 = np.min([x[1]['s'],x[1]['b'][0],x[1]['r']])
        x[0]['s'] = x[0]['s']-m1+Dv[0][t]+act[0]['w']
        x[1]['s'] = x[1]['s']-m2+Dv[1][t]+act[1]['w']
        x[0]['b'][0] = x[0]['b'][0]-m1+act[0]['q']+x[0]['b'][1]+B[0]*(t==0)
        x[1]['b'][0] = x[1]['b'][0]-m2+act[1]['q']+x[1]['b'][1]+B[1]*(t==0)
        for i in np.arange(1,L-1):
            x[0]['b'][i] = x[0]['b'][i+1]
            x[1]['b'][i] = x[1]['b'][i+1]
        x[0]['b'][L-1] = m1
        x[1]['b'][L-1] = m2
        x[0]['r'] = x[0]['r']-m1+act[0]['a']+act[0]['e']
        x[1]['r'] = x[1]['r']-m2+act[1]['a']+act[1]['e']
        
        bsv1[t] = np.max([x[0]['s']-x[0]['b'][0],0])
        bsv2[t] = np.max([x[1]['s']-x[1]['b'][0],0])
        rsv1[t] = np.max([x[0]['s']-x[0]['r'],0])
        rsv2[t] = np.max([x[1]['s']-x[1]['r'],0])
        
        bov1[t] = np.max([-x[0]['s']+x[0]['b'][0],0])
        bov2[t] = np.max([-x[1]['s']+x[1]['b'][0],0])
        rov1[t] = np.max([-x[0]['s']+x[0]['r'],0])
        rov2[t] = np.max([-x[1]['s']+x[1]['r'],0])
        
        t = t+1
    
    sim_res[0] = {'B':Bv1,'s':sv1,'b0':b0v1,'r':rv1,'a':av1,'q':qv1,'w':wv1,'e':ev1,'bs':bsv1,'rs':rsv1,'bo':bov1,'ro':rov1}
    sim_res[1] = {'B':Bv2,'s':sv2,'b0':b0v2,'r':rv2,'a':av2,'q':qv2,'w':wv2,'e':ev2,'bs':bsv2,'rs':rsv2,'bo':bov2,'ro':rov2}

    return sim_res

# In[]:
def test_co(d,L,T,x,D,A,B,method,kwargs):
    # kwargs: paras, p, beta, cL, Zdict
    test_res = {}
    
    N = np.shape(D[0])[0]
    T = np.shape(D[0])[1]
    Bm1 = np.zeros((N,T))
    sm1 = np.zeros((N,T))
    b0m1 = np.zeros((N,T))
    rm1 = np.zeros((N,T))
    
    am1 = np.zeros((N,T))
    qm1 = np.zeros((N,T))
    wm1 = np.zeros((N,T))
    em1 = np.zeros((N,T))
    
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
    wm2 = np.zeros((N,T))
    em2 = np.zeros((N,T))
    
    bsm2 = np.zeros((N,T))
    rsm2 = np.zeros((N,T))
    bom2 = np.zeros((N,T))
    rom2 = np.zeros((N,T))
    
    for n in range(N):
        xtemp = copy.deepcopy(x)
        sim_res = sim_co(d, L, T, xtemp, {0:D[0][n],1:D[1][n]}, {0:A[0][n],1:A[1][n]}, B, method, kwargs)
        Bm1[n], sm1[n], b0m1[n], rm1[n], am1[n], qm1[n], wm1[n], em1[n], bsm1[n], rsm1[n], bom1[n], rom1[n] = sim_res[0]['B'], sim_res[0]['s'], sim_res[0]['b0'],sim_res[0]['r'],sim_res[0]['a'],sim_res[0]['q'],sim_res[0]['w'],sim_res[0]['e'],sim_res[0]['bs'],sim_res[0]['rs'],sim_res[0]['bo'],sim_res[0]['ro']
        Bm2[n], sm2[n], b0m2[n], rm2[n], am2[n], qm2[n], wm2[n], em2[n], bsm2[n], rsm2[n], bom2[n], rom2[n] = sim_res[1]['B'], sim_res[1]['s'], sim_res[1]['b0'],sim_res[1]['r'],sim_res[1]['a'],sim_res[1]['q'],sim_res[1]['w'],sim_res[1]['e'],sim_res[1]['bs'],sim_res[1]['rs'],sim_res[1]['bo'],sim_res[1]['ro']
    
    test_res[0] = {'B':Bm1,'s':sm1,'b0':b0m1,'r':rm1,'a':am1,'q':qm1,'w':wm1,'e':em1,'bs':bsm1,'rs':rsm1,'bo':bom1,'ro':rom1}    
    test_res[1] = {'B':Bm2,'s':sm2,'b0':b0m2,'r':rm2,'a':am2,'q':qm2,'w':wm2,'e':em2,'bs':bsm2,'rs':rsm2,'bo':bom2,'ro':rom2}
    
    return test_res

# In[]:
def cost_co(test_res,paras,beta):
    cR, hR, pR, cB, hB, pB, KS, KR, KB = paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6], paras[7], paras[8]
    am1, qm1, wm1, em1, bom1, rom1, bsm1, rsm1 = test_res[0]['a'], test_res[0]['q'], test_res[0]['w'], test_res[0]['e'], test_res[0]['bo'], test_res[0]['ro'], test_res[0]['bs'], test_res[0]['rs']
    am2, qm2, wm2, em2, bom2, rom2, bsm2, rsm2 = test_res[1]['a'], test_res[1]['q'], test_res[1]['w'], test_res[1]['e'], test_res[1]['bo'], test_res[1]['ro'], test_res[1]['bs'], test_res[1]['rs']
    N = np.shape(am1)[0]
    T = np.shape(am1)[1]
    betaT =np.matrix([beta**t for t in range(T)])
    oneN = np.transpose(np.matrix(np.ones(N)))
    betam = np.dot(oneN,betaT)
    
    costm1 = cB*qm1+cR*am1+hB*bom1+hR*rom1+pB*bsm1+pR*rsm1+KS*0.5*np.absolute(wm1)+KR*0.5*np.absolute(em1)+KB*0.5*np.absolute(qm1)
    costdm1 = np.multiply(betam,costm1)
    costdv1 = np.sum(costdm1,axis=1)
    ec1 = np.sum(costdv1)/N
    
    costm2 = cB*qm2+cR*am2+hB*bom2+hR*rom2+pB*bsm2+pR*rsm2+KS*0.5*np.absolute(wm2)+KR*0.5*np.absolute(em2)+KB*0.5*np.absolute(qm2)
    costdm2 = np.multiply(betam,costm2)
    costdv2 = np.sum(costdm2,axis=1)
    ec2 = np.sum(costdv2)/N
    return ec1,np.squeeze(np.asarray(costdv1)),ec2,np.squeeze(np.asarray(costdv2))

# In[]:
def solve_co(d,L,T,x,B,Dtr,Atr,paras,p,beta,bmax,method,Num):
    costm = np.zeros((bmax,bmax))
    indm = np.zeros((bmax,bmax))
    
    if method=='myo':
        kwargs = {'paras':paras}
    if method=='emyo':
        Zdict = get_Zdict(d, p, paras, beta)
        kwargs = {'paras':paras,'Zdict':Zdict}
    if method=='mdl1':
        kwargs = {'paras':paras,'p':p,'beta':beta}
    if method=='mdl2':
        kwargs = {'paras':paras,'p':p,'beta':beta}
    
    test_res = test_co(d, L, T, x, Dtr, Atr, B, method, kwargs)
    ec1, _, ec2, _ = cost_co(test_res, paras, beta)
    c0 = ec1+ec2
    costm[B[0],B[1]] = c0
    indm[B[0],B[1]] = 1
    
    search = True
    num = 0
    
    while search:
        num = num+1
        # search facility 1
        if B[0]>0:
            B1 = [B[0]-1,B[1]]
            if indm[B1[0],B1[1]]==0:
                test_res1 = test_co(d, L, T, x, Dtr, Atr, B1, method, kwargs)
                ec1, _, ec2, _ = cost_co(test_res1, paras, beta)
                costm[B1[0],B1[1]]=ec1+ec2
                indm[B1[0],B1[1]]=1
            c1 = costm[B1[0],B1[1]]
        else:
            c1 = np.inf
        if B[0]<bmax:
            B2 = [B[0]+1,B[1]]
            if indm[B2[0],B2[1]]==0:
                test_res2 = test_co(d, L, T, x, Dtr, Atr, B2, method, kwargs)
                ec1, _, ec2, _ = cost_co(test_res2, paras, beta)
                costm[B2[0],B2[1]]=ec1+ec2
                indm[B2[0],B2[1]]=1
            c2 = costm[B2[0],B2[1]]
        else:
            c2 = np.inf
        # search facility 2
        if B[1]>0:
            B3 = [B[0],B[1]-1]
            if indm[B3[0],B3[1]]==0:
                test_res3 = test_co(d, L, T, x, Dtr, Atr, B3, method, kwargs)
                ec1, _, ec2, _ = cost_co(test_res3, paras, beta)
                costm[B3[0],B3[1]]=ec1+ec2
                indm[B3[0],B3[1]]=1
            c3 = costm[B3[0],B3[1]]
        else:
            c3 = np.inf
        if B[1]<bmax:
            B4 = [B[0],B[1]+1]
            if indm[B4[0],B4[1]]==0:
                test_res4 = test_co(d, L, T, x, Dtr, Atr, B4, method, kwargs)
                ec1, _, ec2, _ = cost_co(test_res4, paras, beta)
                costm[B4[0],B4[1]]=ec1+ec2
                indm[B4[0],B4[1]]=1
            c4 = costm[B4[0],B4[1]]
        else:
            c4 = np.inf
        c_nb = np.array([c0,c1,c2,c3,c4])
        B_nb = np.array([B,B1,B2,B3,B4])
        cmin_ind = np.argmin(c_nb)
        if cmin_ind == 0:
            search = False
        if num > Num:
            search = False
        B = B_nb[cmin_ind]
        c0 = c_nb[cmin_ind]
        print(B)
    
    return B

def solve_co_syn(d,L,T,x,B,Dtr,Atr,paras,p,beta,bmax,method,Num):
    costv = np.zeros(bmax)
    indv = np.zeros(bmax)
    Bs = B[0]
    
    if method=='myo':
        kwargs = {'paras':paras}
    if method=='emyo':
        Zdict = get_Zdict(d, p, paras, beta)
        kwargs = {'paras':paras,'Zdict':Zdict}
    if method=='mdl1':
        kwargs = {'paras':paras,'p':p,'beta':beta}
    if method=='mdl2':
        kwargs = {'paras':paras,'p':p,'beta':beta}
    
    test_res = test_co(d, L, T, x, Dtr, Atr, [Bs,Bs], method, kwargs)
    ec1, _, ec2, _ = cost_co(test_res, paras, beta)
    c0 = ec1+ec2
    costv[Bs] = c0
    indv[Bs] = 1
    
    search = True
    num = 0
    
    while search:
        num = num+1
        
        if Bs>0:
            Bs1= Bs-1
            if indv[Bs1] == 0:
                test_res1 = test_co(d, L, T, x, Dtr, Atr, [Bs1,Bs1], method, kwargs)
                ec1, _, ec2, _ = cost_co(test_res1, paras, beta)
                costv[Bs1] = ec1+ec2
                indv[Bs1] = 1
            c1 = costv[Bs1]
        else:
            c1 = np.inf
        if Bs<bmax:
            Bs2 = Bs+1
            if indv[Bs2] == 0:
                test_res2 = test_co(d, L, T, x, Dtr, Atr, [Bs2,Bs2], method, kwargs)
                ec1, _, ec2, _ = cost_co(test_res2, paras, beta)
                costv[Bs2] = ec1+ec2
                indv[Bs2] = 1
            c2 = costv[Bs2]
        else:
            c2 = np.inf
        
        c_nb = np.array([c0,c1,c2])
        print(c_nb)
        Bs_nb = np.array([Bs,Bs1,Bs2])
        cmin_ind = np.argmin(c_nb)
        if cmin_ind == 0:
            search = False
        if num > Num:
            search = False
        Bs = Bs_nb[cmin_ind]
        c0 = c_nb[cmin_ind]
        print(Bs)
    
    return [Bs,Bs]


# In[]:
def get_act_myo(d,x,At,paras):
    act = {}
    
    s1, b1, r1 = x[0]['s'],x[0]['b'],x[0]['r']
    s2, b2, r2 = x[1]['s'],x[1]['b'],x[1]['r']
    m1 = np.min([s1,b1[0],r1])
    m2 = np.min([s2,b2[0],r2])
    d_max = int(poisson.ppf(0.99,d[0]))
    pdv = poisson.pmf(np.arange(d_max+1),d[0])
    cR, hR, pR, hB, pB, KS, KR, KB = paras[0], paras[1], paras[2], paras[4], paras[5], paras[6], paras[7], paras[8]
    A1 = At[0]
    A2 = At[1]
    
    M_myo = gp.Model('get-act-myo')
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
    
    if A1 == 1:
        M_myo.addConstr(a[0]==0)
    if A2 == 1:
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
    M_myo.addConstrs((Srd[1,i]+r2+a[1]+e[1]-s2-i-w[1]>=0 for i in range(d_max+1)))
    
    M_myo.addConstrs((Obd[0,i]-b1[0]-b1[1]-q[0]+s1+i+w[0]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Obd[1,i]-b2[0]-b2[1]-q[1]+s2+i+w[1]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Sbd[0,i]+b1[0]+b1[1]+q[0]-s1-i-w[0]>=0 for i in range(d_max+1)))
    M_myo.addConstrs((Sbd[1,i]+b2[0]+b2[1]+q[1]-s2-i-w[1]>=0 for i in range(d_max+1)))
    
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
    
    act[0] = {'a':a1,'q':q1,'w':w1,'e':e1}
    act[1] = {'a':a2,'q':q2,'w':w2,'e':e2}
    
    return act

# In[]:
# def get_Zdict(d,p,paras,beta):
#     cR, hR, pR = paras[0], paras[1], paras[2]
#     rhoR = (pR-(1-beta)*cR)/(pR+hR)
#     Zdict = {}
#     for j in range(2):
#         Ztemp = np.zeros(30)
#         z = poisson.ppf(rhoR,d[j])
#         for k in range(30):
#             wv = np.array([p[j]**i*beta**i for i in range(k+1)])
#             wv[0] = 1
#             funv = np.array([-pR+(pR+hR)*poisson.cdf(z,d[j]*(i+1)) for i in range(k+1)])
#             phi = (1-beta)*cR+np.dot(wv,funv)
#             while phi<0:
#                 z = z+1
#                 funv = np.array([-pR+(pR+hR)*poisson.cdf(z,d[j]*(i+1)) for i in range(k+1)])
#                 phi = (1-beta)*cR+np.dot(wv,funv)
#             Ztemp[k] = z
#         Zdict[j] = np.copy(Ztemp)
#     return Zdict

def get_act_emyo(d,x,At,t,T,paras,Zdict):
    act = {}
    
    s1, b1, r1 = x[0]['s'],x[0]['b'],x[0]['r']
    s2, b2, r2 = x[1]['s'],x[1]['b'],x[1]['r']
    m1 = np.min([s1,b1[0],r1])
    m2 = np.min([s2,b2[0],r2])
    d_max = int(poisson.ppf(0.99,d[0]))
    pdv = poisson.pmf(np.arange(d_max+1),d[0])
    hR, pR, hB, pB, KS, KR, KB = paras[1], paras[2], paras[4], paras[5], paras[6], paras[7], paras[8]
    
    M_emyo = gp.Model('get-act-emyo')
    a = np.zeros(2)
    for i in range(2):
        if At[i] == 0:
            Z = Zdict[i]
            a[i] = max([0,x[i]['s']-x[i]['r']+Z[min([29,T-t])]])   
    q = M_emyo.addVars(2,obj=0,vtype=GRB.INTEGER,name='q')
    qq = M_emyo.addVars(2,obj=KB/2,vtype=GRB.CONTINUOUS,name='qq')
    w = M_emyo.addVars(2,obj=0,vtype=GRB.INTEGER,name='w')
    ww = M_emyo.addVars(2,obj=KS/2,vtype=GRB.CONTINUOUS,name='ww')
    e = M_emyo.addVars(2,obj=0,vtype=GRB.INTEGER,name='e')
    ee = M_emyo.addVars(2,obj=KR/2,vtype=GRB.CONTINUOUS,name='ee')
    Ord = M_emyo.addVars(2,d_max+1,obj=hR*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Ord')
    Srd = M_emyo.addVars(2,d_max+1,obj=pR*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Srd')
    Obd = M_emyo.addVars(2,d_max+1,obj=hB*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Obd')
    Sbd = M_emyo.addVars(2,d_max+1,obj=pB*np.array([pdv,pdv]),lb=0,vtype=GRB.CONTINUOUS,name='Sbd')
    
    M_emyo.addConstrs((qq[i]-q[i]>=0 for i in range(2)))
    M_emyo.addConstrs((qq[i]+q[i]>=0 for i in range(2)))
    M_emyo.addConstrs((ww[i]-w[i]>=0 for i in range(2)))
    M_emyo.addConstrs((ww[i]+w[i]>=0 for i in range(2)))
    M_emyo.addConstrs((ee[i]-e[i]>=0 for i in range(2)))
    M_emyo.addConstrs((ee[i]+e[i]>=0 for i in range(2)))
    
    M_emyo.addConstr(q[0]+q[1]==0)
    M_emyo.addConstr(w[0]+w[1]==0)
    M_emyo.addConstr(e[0]+e[1]==0)
    M_emyo.addConstr(w[0]+s1-m1>=0)
    M_emyo.addConstr(w[1]+s2-m2>=0)
    M_emyo.addConstr(e[0]+a[0]+r1-m1>=0)
    M_emyo.addConstr(e[1]+a[1]+r2-m2>=0)
    M_emyo.addConstr(q[0]+b1[0]+b1[1]-m1>=0)
    M_emyo.addConstr(q[1]+b2[0]+b2[1]-m2>=0)
    
    M_emyo.addConstrs((Ord[0,i]-r1-a[0]-e[0]+s1+i+w[0]>=0 for i in range(d_max+1)))
    M_emyo.addConstrs((Ord[1,i]-r2-a[1]-e[1]+s2+i+w[1]>=0 for i in range(d_max+1)))
    M_emyo.addConstrs((Srd[0,i]+r1+a[0]+e[0]-s1-i-w[0]>=0 for i in range(d_max+1)))
    M_emyo.addConstrs((Srd[1,i]+r2+a[1]+e[1]-s2-i-w[1]>=0 for i in range(d_max+1)))
    
    M_emyo.addConstrs((Obd[0,i]-b1[0]-b1[1]-q[0]+s1+i+w[0]>=0 for i in range(d_max+1)))
    M_emyo.addConstrs((Obd[1,i]-b2[0]-b2[1]-q[1]+s2+i+w[1]>=0 for i in range(d_max+1)))
    M_emyo.addConstrs((Sbd[0,i]+b1[0]+b1[1]+q[0]-s1-i-w[0]>=0 for i in range(d_max+1)))
    M_emyo.addConstrs((Sbd[1,i]+b2[0]+b2[1]+q[1]-s2-i-w[1]>=0 for i in range(d_max+1)))
    
    M_emyo.setParam('OutputFlag',False)
    M_emyo.optimize()
    
    q1 = q[0].X
    q2 = q[1].X
    w1 = w[0].X
    w2 = w[1].X
    e1 = e[0].X
    e2 = e[1].X
    
    act[0] = {'a':a[0],'q':q1,'w':w1,'e':e1}
    act[1] = {'a':a[1],'q':q2,'w':w2,'e':e2}
    
    return act

# In[]:
def get_act_mdl(d,x,At,t,T,paras,cL,p,beta):
    act = {}
    s1, b1, r1 = x[0]['s'],x[0]['b'],x[0]['r']
    s2, b2, r2 = x[1]['s'],x[1]['b'],x[1]['r']
    m1 = np.min([s1,b1[0],r1])
    m2 = np.min([s2,b2[0],r2])
    d_max = int(poisson.ppf(0.99,d[0]))
    pdv = poisson.pmf(np.arange(d_max+1),d[0])
    cR, hR, pR, hB, pB, KS, KR, KB = paras[0], paras[1], paras[2], paras[4], paras[5], paras[6], paras[7], paras[8]
    A1 = At[0]
    A2 = At[1]
    
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
    M_mdl.addConstrs((Srd0[1,i]+r2+a0[1]+e0[1]-s2-i-w0[1]>=0 for i in range(d_max+1)))
    
    M_mdl.addConstrs((Obd0[0,i]-b1[0]-b1[1]-q0[0]+s1+i+w0[0]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Obd0[1,i]-b2[0]-b2[1]-q0[1]+s2+i+w0[1]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Sbd0[0,i]+b1[0]+b1[1]+q0[0]-s1-i-w0[0]>=0 for i in range(d_max+1)))
    M_mdl.addConstrs((Sbd0[1,i]+b2[0]+b2[1]+q0[1]-s2-i-w0[1]>=0 for i in range(d_max+1)))
    
    # period 1:
    if cL>0:
        pAm = np.array([[1-p[0],p[0]],[1-p[1],p[1]]])
        a1 = M_mdl.addVars(2,2,obj=beta*cR*pAm,lb=0,vtype=GRB.INTEGER,name='a1')
        q1 = M_mdl.addVars(2,2,obj=0,vtype=GRB.INTEGER,name='q1')
        qq1 = M_mdl.addVars(2,2,obj=beta*KB/2*pAm,vtype=GRB.CONTINUOUS,name='qq1')
        w1 = M_mdl.addVars(2,2,obj=0,vtype=GRB.INTEGER,name='w1')
        ww1 = M_mdl.addVars(2,2,obj=beta*KS/2*pAm,vtype=GRB.CONTINUOUS,name='ww1')
        e1 = M_mdl.addVars(2,2,obj=0,vtype=GRB.INTEGER,name='e1')
        ee1 = M_mdl.addVars(2,2,obj=beta*KR/2*pAm,vtype=GRB.CONTINUOUS,name='ee1')
        pAdv11 = (1-p[0])*pdv
        pAdv12 = p[0]*pdv
        pAdm1 = np.array([pAdv11,pAdv12])
        pAdv21 = (1-p[1])*pdv
        pAdv22 = p[1]*pdv
        pAdm2 = np.array([pAdv21,pAdv22])
        pAdt = np.array([pAdm1,pAdm2])
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
        pAAm = np.array([[(1-p[0])**2,(1-p[0])*p,p[0]*(1-p[0]),p[0]**2],[(1-p[1])**2,(1-p[1])*p[1],p[1]*(1-p[1]),p[1]**2]])
        a2 = M_mdl.addVars(2,4,obj=beta**2*cR*pAAm,vtype=GRB.INTEGER,name='a2')
        q2 = M_mdl.addVars(2,4,obj=0,vtype=GRB.INTEGER,name='q2')
        qq2 = M_mdl.addVars(2,4,obj=beta**2*KB/2*pAAm,vtype=GRB.CONTINUOUS,name='qq2')
        w2 = M_mdl.addVars(2,4,obj=0,vtype=GRB.INTEGER,name='w2')
        ww2 = M_mdl.addVars(2,4,obj=beta**2*KS/2*pAAm,vtype=GRB.CONTINUOUS,name='ww2')
        e2 = M_mdl.addVars(2,4,obj=0,vtype=GRB.INTEGER,name='e2')
        ee2 = M_mdl.addVars(2,4,obj=beta**2*KR/2*pAAm,vtype=GRB.CONTINUOUS,name='ee2')
        pAAdm1 = np.array([(1-p[0])**2*pdv,(1-p[0])*p[0]*pdv,p[0]*(1-p[0])*pdv,p[0]**2*pdv])
        pAAdm2 = np.array([(1-p[1])**2*pdv,(1-p[1])*p[1]*pdv,p[1]*(1-p[1])*pdv,p[1]**2*pdv])
        pAAdt = np.array([pAAdm1,pAAdm2])
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
    
    act[0] = {'a':a1o,'q':q1o,'w':w1o,'e':e1o}
    act[1] = {'a':a2o,'q':q2o,'w':w2o,'e':e2o}
    
    return act