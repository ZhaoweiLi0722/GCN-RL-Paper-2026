#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 01:06:27 2020
Reference1: https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
Reference2: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
@author: howard
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        """A Capacity Planning Environment for OpenAI gym"""
        # define initial resource pool in Fab1 and Fab2
        #self.env = gym.make("foo-v0")
        # planning horizon = 104 weeks
        # weekly demand rate poisson 250/52
        # Discount factor = 0.9
        # Disruption prob. = 0.3 (Supplier supply)
        
        self.L = 3 # length of production
              
        self.minDemand_1 = 0
        self.maxDemand_1 = 250/52
        
        self.initSpeci_1 = 0
        self.minSpeci_1 = 0
        self.maxSpeci_1 = 2000

        
        self.iniBio_1 = 23
        self.minBio_1 = 0
        self.maxBio_1 = 100

        
        self.initReag_1 = 0
        self.minReag_1 = 0
        self.maxReag_1 = 2000

        
        self.minReagRplen_1 = 0
        self.maxReagRplen_1 = 100
        
        self.reagAvailable_1 = 0.7

        
        self.minDemand_2 = 0
        self.maxDemand_2 = 250/52

        
        self.initSpeci_2 = 0
        self.minSpeci_2 = 0
        self.maxSpeci_2 = 2000

        
        self.iniBio_2 = 23
        self.minBio_2 = 0
        self.maxBio_2 = 100

        
        self.initReag_2 = 0
        self.minReag_2 = 0
        self.maxReag_2 = 2000

        
        self.minReagRplen_2 = 0
        self.maxReagRplen_2 = 100
        
        self.reagAvailable_2 = 0.7
        
        self.init_bio = np.array([self.iniBio_1, self.minBio_1, self.minBio_1, self.iniBio_2, self.minBio_2, self.minBio_2])
        #self.state = np.array([np.random.poisson(self.maxDemand_1), self.initSpeci_1, self.initReag_1, np.random.binomial(1, self.reagAvailable_1), np.random.poisson(self.maxDemand_2), self.initSpeci_2, self.initReag_2, np.random.binomial(1, self.reagAvailable_2)])
        self.init_state = np.array([self.minDemand_1, self.initSpeci_1, self.initReag_1, np.random.binomial(1, self.reagAvailable_1), self.minDemand_2, self.initSpeci_2, self.initReag_2, np.random.binomial(1, self.reagAvailable_2)])
        self.init_action = np.array([self.minSpeci_1,self.minBio_1,self.minReag_1,self.minReagRplen_1,self.minReagRplen_2], dtype=np.float32)
        self.state = np.append(self.init_state, self.init_bio)
        self.state = np.append(self.state, self.init_action)
        
        self.state_eval = self.state

        
        self.bio_low_state = np.array([self.minBio_1, self.minBio_1, self.minBio_1, self.minBio_2, self.minBio_2, self.minBio_2])
        self.bio_high_state = np.array([self.maxBio_1, self.maxBio_1, self.maxBio_1, self.maxBio_2, self.maxBio_2, self.maxBio_2])
        self.low_state = np.array([self.minDemand_1, self.minSpeci_1, self.minReag_1, 0, self.minDemand_2, self.minSpeci_2, self.minReag_2, 0])
        self.high_state = np.array([self.maxDemand_1, self.maxSpeci_1, self.maxReag_1, 1,self.maxDemand_1, self.maxSpeci_2,self.maxReag_2, 1])
        self.low_action = np.array([-self.maxSpeci_1, -self.maxBio_1, -self.maxReag_1, self.minReagRplen_1, self.minReagRplen_2])
        self.high_action = np.array([self.maxSpeci_2, self.maxBio_2, self.maxReag_2,self.maxReagRplen_1, self.maxReagRplen_2])
        self.low_state = np.append(self.low_state, self.bio_low_state)
        self.low_state = np.append(self.low_state, self.low_action)
        self.high_state = np.append(self.high_state, self.bio_high_state)
        self.high_state = np.append(self.high_state, self.high_action)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, 0, 0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        
        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        bio_state_temp = np.array([self.state[8],self.state[9],self.state[10],self.state[11],self.state[12],self.state[13]])
        speci_1 = self.state[1]
        reag_1 = self.state[2]
        reagAvai_1 = self.state[3]
        bio_1 = self.state[8]
        
        speci_2 = self.state[5]
        reag_2 = self.state[6]
        reagAvai_2 = self.state[7]
        bio_2 = self.state[11]
        
        
        ### Action should be integer 
        #action_norm = np.clip(action, -1, 1)
        speciTrans = action[0] * 20
        bioTrans = action[1] * 10
        reagTrans = action[2] * 20
        reagReplen_1 = (action[3]) * 20 * reagAvai_1
        reagReplen_2 = (action[4]) * 20 * reagAvai_2
        
        speciTrans = round(speciTrans)
        bioTrans = round(bioTrans)
        reagTrans = round(reagTrans)
        reagReplen_1 = round(reagReplen_1)
        reagReplen_2 = round(reagReplen_2)
        
            
        # System dynamic of fab1 and fab2
        # Specimen transition
 
        if speciTrans > 0 and abs(speciTrans) > speci_2:
            speciTrans = speci_2
        if speciTrans < 0 and abs(speciTrans) > speci_1:
            speciTrans = speci_1            
        speci_1 = np.clip(speci_1 + speciTrans, self.minSpeci_1, self.maxSpeci_1)
        speci_2 = np.clip(speci_2 - speciTrans, self.minSpeci_2, self.maxSpeci_2)
        
        demand_1 = np.random.poisson(self.maxDemand_1)
        demand_2 = np.random.poisson(self.maxDemand_2)
        speci_1 = np.clip(speci_1 + demand_1, self.minSpeci_1, self.maxSpeci_1)
        speci_2 = np.clip(speci_2 + demand_2, self.minSpeci_2, self.maxSpeci_2)      

        # Reagent transition
        if reagTrans > 0 and abs(reagTrans) > reag_2:
            reagTrans = reag_2
            
        if reagTrans < 0 and abs(reagTrans) > reag_1:
            reagTrans = reag_1
        
        reag_1 = np.clip(reag_1 + reagTrans, self.minReag_1, self.maxReag_1)
        reag_2 = np.clip(reag_2  - reagTrans, self.minReag_2, self.maxReag_2)
        
        if reagReplen_1 > 0:
            reag_1 = np.clip(reag_1 + reagReplen_1, self.minReag_1, self.maxReag_1)
        else:
            reagReplen_1 = 0
            
        if reagReplen_2 > 0:
            reag_2 = np.clip(reag_2 + reagReplen_2, self.minReag_2, self.maxReag_2)
        else:
            reagReplen_2 = 0
        
        # Bioreactor transition
        
        if bioTrans > 0 and abs(bioTrans) > bio_2:
            bioTrans = bio_2
        if bioTrans < 0 and abs(bioTrans) > bio_1:
            bioTrans = bio_1
        
        bio_1 = np.clip(bio_1 + bio_state_temp[1] + bioTrans, self.minBio_1, self.maxBio_1)
        bio_2 = np.clip(bio_2 + bio_state_temp[4] - bioTrans, self.minBio_2, self.maxBio_2)

        
        m1 = np.min([speci_1,bio_1,reag_1])
        if reag_1 < speci_1:
            under_reag_1 = speci_1 - reag_1
            idle_reag_1 = 0
        else:
            under_reag_1 = 0
            idle_reag_1 = reag_1 - speci_1
            
        if bio_1 < speci_1:
            under_bio_1 = speci_1 - bio_1
            idle_bio_1 = 0
        else:
            under_bio_1 = 0
            idle_bio_1 = bio_1 - speci_1
            
        speci_1 = speci_1 - m1
        reag_1 = reag_1 - m1
        bio_state_temp[0] = bio_1 - m1
        bio_state_temp[1] = bio_state_temp[2]
        bio_state_temp[2] = m1
        
        speci_idle_1 = speci_1
        speci_proc_1 = m1 + bio_state_temp[1]
        
        reag__idle_1 = reag_1
        speci_proc_1 = m1
        
        # state transit
        #m1 = np.min([s1,b1[0],r1])
        #s1 = s1-m1+D1+w1
        #b1[0] = b1[0]-m1+q1+b1[1]+B1
        #for i in np.arange(1,L-1):
            #b1[i] = b1[i+1]
            #b1[L-1] = m1
            #r1 = r1-m1+a1+e1
        
        m2 = np.min([speci_2,bio_2,reag_2])
        if reag_2 < speci_2:
            under_reag_2 = speci_2 - reag_2
            idle_reag_2 = 0
        else:
            under_reag_2 = 0
            idle_reag_2 = reag_2 - speci_2
            
        if bio_2 < speci_2:
            under_bio_2 = speci_2 - bio_2
            idle_bio_2 = 0
        else:
            under_bio_2 = 0
            idle_bio_2 = bio_2 - speci_2
            
        speci_2 = speci_2 - m2
        reag_2 = reag_2 - m2
        bio_state_temp[3] = bio_2 - m2
        bio_state_temp[4] = bio_state_temp[5]
        bio_state_temp[5] = m2
        
        speci_idle_2 = speci_2
        speci_proc_2 = m2 + bio_state_temp[4]
        
        reag__idle_2 = reag_2
        speci_proc_2 = m2
        
        
        #self.state = np.array([np.random.poisson(self.maxDemand_1), speci_1, reag_1, np.random.binomial(1, self.reagAvailable_1), np.random.poisson(self.maxDemand_2), speci_2, reag_2, np.random.binomial(1, self.reagAvailable_2)])
        state_temp = np.array([demand_1,speci_1, reag_1, np.random.binomial(1, self.reagAvailable_1),demand_2, speci_2, reag_2, np.random.binomial(1, self.reagAvailable_2)])
        state_temp = np.append(state_temp,bio_state_temp)
        action_acctual = np.array([speciTrans,bioTrans,reagTrans,reagReplen_1,reagReplen_2])
        state_temp = np.append(state_temp,action_acctual)

        self.state = state_temp
        
        temp = np.array([speci_idle_1,speci_idle_2,speci_proc_1,speci_proc_2])
        self.state_eval = np.append(state_temp,temp)
        
        # calculate reward
        # paras = [reagent purchase cost 42174, reagent holding cost 113.5, reagent under stock penalty 86504.5, bioreactor purchase cost 25000, idle bioreactor holding cost 14.4, bioreactor understock penalty 50273.6, speci trans cost 600, bio trans cost 500, reag trans cost 200]
        r = 0
        if reagReplen_1 > 0:
            r += reagReplen_1
        if reagReplen_2 > 0:
            r += reagReplen_2
        
        c = [r, idle_reag_1+idle_reag_2, under_reag_1+under_reag_2, idle_bio_1+idle_bio_2, under_bio_1+under_bio_2, abs(speciTrans), abs(bioTrans), abs(reagTrans)]
        paras = [42174, 113.5, 86504.5*1.4, 14.4, 50273.6*1.4, 600, 1000, 200]
        #paras = [0, 113.5, 86504.5*1.4, 14.4, 50273.6*1.4, 600, 1000, 200]
        norm = np.linalg.norm(paras)
        paras_norm = paras/norm
        costs = np.dot(c, paras_norm)
        #costs = np.dot(c, paras)
        return self._get_obs(), -costs, False, {}, self._get_state()
        
    def reset(self):
        self.init_bio = np.array([self.iniBio_1, self.minBio_1, self.minBio_1, self.iniBio_2, self.minBio_2, self.minBio_2])
        #self.state = np.array([np.random.poisson(self.maxDemand_1), self.initSpeci_1, self.initReag_1, np.random.binomial(1, self.reagAvailable_1), np.random.poisson(self.maxDemand_2), self.initSpeci_2, self.initReag_2, np.random.binomial(1, self.reagAvailable_2)])
        self.init_state = np.array([self.minDemand_1, self.initSpeci_1, self.initReag_1, np.random.binomial(1, self.reagAvailable_1), self.minDemand_2, self.initSpeci_2, self.initReag_2, np.random.binomial(1, self.reagAvailable_2)])
        self.init_action = np.array([self.minSpeci_1,self.minSpeci_1,self.minSpeci_1,self.minSpeci_1,self.minSpeci_1], dtype=np.float32)
        self.state = np.append(self.init_state, self.init_bio)
        self.state = np.append(self.state, self.init_action)
        return self._get_obs()
    
    def _get_obs(self):
        obs_norm = self.state / np.array([self.maxDemand_1, self.maxSpeci_1,self.maxReag_1,1,self.maxDemand_2,self.maxSpeci_2,self.maxReag_2,1,self.maxBio_1,self.maxBio_1,self.maxBio_1,self.maxBio_2,self.maxBio_2,self.maxBio_2,20,10,20,20,20])
        return obs_norm
    
    def _get_state(self):
        obs = self.state_eval #/ np.array([self.maxDemand_1, self.maxSpeci_1,self.maxReag_1,1,self.maxDemand_2,self.maxSpeci_2,self.maxReag_2,1,self.maxBio_1,self.maxBio_1,self.maxBio_1,self.maxBio_2,self.maxBio_2,self.maxBio_2,200,46,200,200,200])
        return obs