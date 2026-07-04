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
        
        self.L = 5 # length of production
        
        self.minDemand_1 = 0
        self.maxDemand_1 = 20
        
        self.initSpeci_1 = 0
        self.minSpeci_1 = 0
        self.maxSpeci_1 = 100
        
        self.iniBio_1 = 10
        self.minBio_1 = 0
        self.maxBio_1 = 20
        
        self.initReag_1 = 100
        self.minReag_1 = 0
        self.maxReag_1 = 200
        
        self.minReagRplen_1 = 0
        self.maxReagRplen_1 = 100
        
        self.minDemand_2 = 0
        self.maxDemand_2 = 20
        
        self.initSpeci_2 = 0
        self.minSpeci_2 = 0
        self.maxSpeci_2 = 100
        
        self.iniBio_2 = 10
        self.minBio_2 = 0
        self.maxBio_2 = 20
        
        self.initReag_2 = 100
        self.minReag_2 = 0
        self.maxReag_2 = 200
        
        self.minReagRplen_2 = 0
        self.maxReagRplen_2 = 100
        
        self.bio_low_state = np.array([self.minBio_1, self.minBio_1, self.minBio_1, self.minBio_1, self.minBio_1, self.minBio_2, self.minBio_2, self.minBio_2, self.minBio_2, self.minBio_2])
        self.bio_high_state = np.array([self.maxBio_1, self.maxBio_1, self.maxBio_1, self.maxBio_1, self.maxBio_1, self.maxBio_2, self.maxBio_2, self.maxBio_2, self.maxBio_2, self.maxBio_2])
        self.low_state = np.array([self.minDemand_1, self.minSpeci_1, self.minReag_1, self.minDemand_2, self.minSpeci_2, self.minReag_2], dtype=np.float32)
        self.high_state = np.array([self.maxDemand_1, self.maxSpeci_1, self.maxReag_1, self.maxDemand_2, self.maxSpeci_2,self.maxReag_2], dtype=np.float32)
        self.low_state = np.append(self.low_state, self.bio_low_state)
        self.high_state = np.append(self.high_state, self.bio_high_state)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        
        self.action_space = spaces.Box(low=np.array([-100.0, -10.0, -100.0, self.minReagRplen_1, self.minReagRplen_2]), high=np.array([100.0, 10.0, 100.0, self.maxReagRplen_1, self.maxReagRplen_2]), dtype=np.float32)
        
        self.seed()
        #self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        demand_1 = self.state[0]
        speci_1 = self.state[1]
        reag_1 = self.state[2]
        demand_2 = self.state[3]
        speci_2 = self.state[4]
        reag_2 = self.state[5]
        bio_1 = self.state[6]
        bio_2 = self.state[11]

        action_norm = np.clip(action, -1, 1)
        speciTrans = np.clip(action_norm[0] * self.maxSpeci_2, -speci_1, speci_2)
        bioTrans = np.clip(action_norm[1] * self.maxBio_2, -bio_1, bio_2)
        reagTrans = np.clip(action_norm[2] * self.maxReag_2, -reag_1, reag_2)
        reagReplen_1 = action_norm[3] * self.maxReagRplen_1
        reagReplen_2 = action_norm[4] * self.maxReagRplen_2
        
        
        
        # System dynamic of fab1 and fab2
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
            
        m1 = np.min([speci_1,bio_1,reag_1])
        speci_1 = np.clip(speci_1 - m1 + demand_1 + speciTrans, self.minSpeci_1, self.maxSpeci_1)
        reag_1 = np.clip(reag_1 - m1 + reagReplen_1 + reagTrans, self.minReag_1, self.maxReag_1)
        self.bio_state[0] = bio_1 - m1 + bioTrans + self.bio_state[1]
        for i in np.arange(1,self.L - 1):
            self.bio_state[i] = self.bio_state[i+1]
        self.bio_state[self.L - 1] = m1
        
        # state transit
        #m1 = np.min([s1,b1[0],r1])
        #s1 = s1-m1+D1+w1
        #b1[0] = b1[0]-m1+q1+b1[1]+B1
        #for i in np.arange(1,L-1):
            #b1[i] = b1[i+1]
            #b1[L-1] = m1
            #r1 = r1-m1+a1+e1
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
            
        m2 = np.min([speci_2,bio_2,reag_2])
        speci_2 = np.clip(speci_2 - m2 + demand_2 - speciTrans, self.minSpeci_2, self.maxSpeci_2)
        reag_2 = np.clip(reag_2 - m1 + reagReplen_2  - reagTrans, self.minReag_2, self.maxReag_2)
        self.bio_state[5] = bio_2 - m2 - bioTrans + self.bio_state[6]
        for i in np.arange(6,self.L - 1 + 5):
            self.bio_state[i] = self.bio_state[i+1]
        self.bio_state[self.L - 1 + 5] = m2
        
        
        self.state = np.array([self.np_random.uniform(low=self.minDemand_1, high=self.maxDemand_1), speci_1, reag_1, self.np_random.uniform(low=self.minDemand_2, high=self.maxDemand_2), speci_2, reag_2])
        self.state = np.append(self.state, self.bio_state)
        
        # calculate reward
        # paras = [reagent purchase cost 42174, reagent holding cost 113.5, reagent under stock penalty 86504.5, bioreactor purchase cost 25000, idle bioreactor holding cost 14.4, bioreactor understock penalty 50273.6, speci trans cost 600, bio trans cost 500, reag trans cost 200]
        c = [reagReplen_1+reagReplen_2, idle_reag_1+idle_reag_2, under_reag_1+under_reag_2, idle_bio_1+idle_bio_2, under_bio_1+under_bio_2, speciTrans, bioTrans, reagTrans]
        paras = [42174, 113.5, 86504.5, 14.4, 50273.6, 600, 500, 200]
        costs = np.dot(c, paras)
        return self.state, -costs, False, {}
        
    def reset(self):
        self.bio_state = np.array([self.iniBio_1, self.minBio_1, self.minBio_1, self.minBio_1, self.minBio_1, self.iniBio_2, self.minBio_2, self.minBio_2, self.minBio_2, self.minBio_2])
        self.state = np.array([self.np_random.uniform(low=self.minDemand_1, high=self.maxDemand_1), self.initSpeci_1, self.initReag_1, self.np_random.uniform(low=self.minDemand_2, high=self.maxDemand_2), self.initSpeci_2, self.initReag_2])
        self.state = np.append(self.state, self.bio_state)
        return np.array(self.state)
    

    #def render(self, mode='human', close=False):