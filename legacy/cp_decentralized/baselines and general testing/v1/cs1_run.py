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
T = 52*2
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
B_iso = solve_iso(d,L,x,Dtr,Atr,paras,p,beta)
Res_iso = test_iso(d,L,x,Dte,Ate,B_iso,paras,p,beta)
ec1_iso,cv1_iso,ec2_iso,cv2_iso = cost_iso(Res_iso,paras,beta)

# In[]: Co-Net myo
B_myo = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'myo',100)
Res_myo = test_co(d,L,T,x,Dte,Ate,B_myo,'myo',{'paras':paras})
ec1_myo,cv1_myo,ec2_myo,cv2_myo = cost_co(Res_myo,paras,beta)

# In[]: Co-Net emyo
B_emyo = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'emyo',100)
Zdict = get_Zdict(d, p, paras, beta)
kwargs = {'paras':paras,'Zdict':Zdict}
Res_emyo = test_co(d,L,T,x,Dte,Ate,B_emyo,'emyo',kwargs)
ec1_emyo,cv1_emyo,ec2_emyo,cv2_emyo = cost_co(Res_emyo,paras,beta)

# In[]: Co-Net mdl1
B_mdl1 = solve_co_syn(d,L,T,x,B_iso,Dtr,Atr,paras,p,beta,100,'mdl1',100)

# In[]: Co-Net mdl2

