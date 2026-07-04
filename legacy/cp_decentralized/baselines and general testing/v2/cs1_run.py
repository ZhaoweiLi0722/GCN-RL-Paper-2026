#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:16:47 2020

@author: junxuanli

Case study 1:
"""

import numpy as np
import matplotlib.pyplot as plt
from cp_rand_gen import *
from cp_isonet_func import *
from cp_conet_func import *

# In[]:
d = [250/52, 250/52] # weekly demand rate
L = 3 # production lead time (L>1)
paras = [42174, 113.5, 86504.5*1.4, 25000, 14.4, 50273.6*1.4, 600, 1000, 200]
# paras = [42174, 113.5, 86504.5*1.4, 25000, 14.4, 50273.6*1.4, 0, 0, 0]
T = 52
Ntr = 100
Nte = 500
cor = 'i'
p = [0.3, 0.3]
beta = 0.9

Dtr = gen_D(d,T,Ntr)
Dte = gen_D(d,T,Nte)

Atr = gen_A(p,T,Ntr,cor)
Ate = gen_A(p,T,Nte,cor)

x = {}
x[0] = {'s':0,'b':np.array([0]*L),'r':0}
x[1] = {'s':0,'b':np.array([0]*L),'r':0}

# In[]: Iso-Net
B_iso = solve_iso_syn(d,L,x,Dtr,Atr,paras,p,beta)
# B_iso = solve_iso(d,L,x,Dtr,Atr,paras,p,beta)
Res_iso = test_iso(d,L,x,Dte,Ate,B_iso,paras,p,beta)
ec1_iso,cv1_iso,ec2_iso,cv2_iso = cost_iso(Res_iso,paras,beta)

# In[]: Co-Net myo
B_myo = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'myo',100)
# B_myo = solve_co(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'myo',100)
Res_myo = test_co(d,L,T,x,Dte,Ate,B_myo,'myo',{'paras':paras})
ec1_myo,cv1_myo,ec2_myo,cv2_myo = cost_co(Res_myo,B_myo,paras,beta)

# In[]: Co-Net emyo
B_emyo = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'emyo',100)
# B_emyo = solve_co(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'emyo',100)
Zdict = get_Zdict(d, p, paras, beta)
kwargs = {'paras':paras,'Zdict':Zdict}
Res_emyo = test_co(d,L,T,x,Dte,Ate,B_emyo,'emyo',kwargs)
ec1_emyo,cv1_emyo,ec2_emyo,cv2_emyo = cost_co(Res_emyo,B_emyo,paras,beta)

# In[]: Co-Net mdl1
B_mdl1 = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'mdl1',100)
# B_mdl1 = solve_co(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'mdl1',100)
kwargs = {'paras':paras,'p':p,'beta':beta}
Res_mdl1 = test_co(d,L,T,x,Dte,Ate,B_mdl1,'mdl1',kwargs)
ec1_mdl1,cv1_mdl1,ec2_mdl1,cv2_mdl1 = cost_co(Res_mdl1,B_mdl1,paras,beta)

# In[]: Co-Net mdl2
B_mdl2 = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'mdl2',100)
# B_mdl2 = solve_co(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'mdl2',100)
kwargs = {'paras':paras,'p':p,'beta':beta}
Res_mdl2 = test_co(d,L,T,x,Dte,Ate,B_mdl2,'mdl2',kwargs)
ec1_mdl2,cv1_mdl2,ec2_mdl2,cv2_mdl2 = cost_co(Res_mdl2,B_mdl2,paras,beta)

# In[]: Co-Net emdl1
B_emdl1 = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'emdl1',100)
# B_emdl1 = solve_co(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'emdl1',100)
kwargs = {'paras':paras,'p':p,'beta':beta,'Zdict':Zdict}
Res_emdl1 = test_co(d,L,T,x,Dte,Ate,B_emdl1,'emdl1',kwargs)
ec1_emdl1,cv1_emdl1,ec2_emdl1,cv2_emdl1 = cost_co(Res_emdl1,B_emdl1,paras,beta)

# In[]: Co-Net emdl2
B_emdl2 = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'emdl2',100)
# B_emdl2 = solve_co(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'emdl2',100)
kwargs = {'paras':paras,'p':p,'beta':beta,'Zdict':Zdict}
Res_emdl2 = test_co(d,L,T,x,Dte,Ate,B_emdl2,'emdl2',kwargs)
ec1_emdl2,cv1_emdl2,ec2_emdl2,cv2_emdl2 = cost_co(Res_emdl2,B_emdl2,paras,beta)

# In[]: LB
B_lb = solve_iso_lb(d,L,x,Dtr,Atr,paras,p,beta)
xx = {'s':x[0]['s']+x[1]['s'],'b':x[0]['b']+x[1]['b'],'r':x[0]['r']+x[1]['r']}
dd = d[0]+d[1]
DD = Dte[0]+Dte[1]
AA = Ate[0]*Ate[1]
pp = p[0]*p[1]
Res_lb = test_iso_lb(dd,L,xx,DD,AA,B_lb,paras,pp,beta)
ec_lb, cv_lb = cost_iso_lb(Res_lb,paras,beta)
