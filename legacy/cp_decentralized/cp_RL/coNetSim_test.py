#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:34:01 2020

@author: howard
"""

import simpy
import numpy as np

def coNet_run(env):    

    # define initial resource pool in Fab1 and Fab2
    initSpeci_1 = 0
    inibio_1 = 10
    initreag_1 = 100
    initSpeci_2 = 0
    inibio_2 = 10
    initreag_2 = 100
    
    # define global variables to track resource in 
    global speci_1, bio_1, reag_1, speci_2, bio_2, reag_2, num_ordered_1, num_ordered_2
    num_ordered_1 = 0
    num_ordered_2 = 0
    speci_1 = initSpeci_1
    bio_1 = inibio_1
    reag_1 = initreag_1
    speci_2 = initSpeci_2
    bio_2 = inibio_2
    reag_2 = initreag_2
    
    while True:
        interarrival = generate_interarrival()
        yield env.timeout(interarrival)
        #balance -= inventory*2*interarrival
        demand_1 = generate_demand_1()
        demand_2 = generate_demand_2()
        speci_1 += demand_1
        speci_2 += demand_2
        available_1 = np.min([bio_1, reag_1])
        available_2 = np.min([bio_2, reag_2])
        
        if speci_1 <= available_1:
            #bio_1 -= speci_1
            reag_1 -= speci_1
            speci_1 -= speci_1
            print('{:.2f} Fab1 processed {}'.format(env.now, demand_1))
        else:
            #bio_1 -= np.min(bio_1, reag_1)
            reag_1 -= available_1
            speci_1 -= available_1
            print('{:.2f} Fab1 processed {} left {}'.format(env.now, available_1, speci_1))
            
        if reag_1 == 0 and num_ordered_1 == 0:
            env.process(handle_order1(env))
            
        if speci_2 <= available_2:
            #bio_2 -= speci_2
            reag_2 -= speci_2
            speci_2 -= speci_2
            print('{:.2f} Fab2 processed {}'.format(env.now, demand_2))
        else:
            #bio_2 -= np.min(bio_2, reag_2)
            reag_2 -= available_2
            speci_2 -= available_2
            print('{:.2f} Fab2 processed {} left {}'.format(env.now, available_2, speci_2))
            
        if reag_2 == 0 and num_ordered_2 == 0:
            env.process(handle_order2(env))

def handle_order1(env):
    global reag_1, num_ordered_1 #inventory, balance, num_ordered
    num_ordered_1 = 100 #order_target - inventory
    #balance -= 50*num_ordered
    print('{:.2f} placed order for {}'.format(env.now, num_ordered_1))
    yield env.timeout(1.0)
    reag_1 += num_ordered_1
    num_ordered_1 = 0
    print('{:.2f} Fab1 received order, {} in inventory'.format(env.now, reag_1))
    
def handle_order2(env):
    global reag_2, num_ordered_2 #inventory, balance, num_ordered
    num_ordered_2 = 100 #order_target - inventory
    #balance -= 50*num_ordered
    print('{:.2f} placed order for {}'.format(env.now, num_ordered_2))
    yield env.timeout(1.0)
    reag_2 += num_ordered_2
    num_ordered_2 = 0
    print('{:.2f} Fab2 received order, {} in inventory'.format(env.now, reag_1))
        
        
def generate_interarrival():
    return np.random.exponential(1./3)
def generate_demand_1():
    return np.random.randint(5, 10)
def generate_demand_2():
    return np.random.randint(1, 10)

obs_time = []
inventory_level_1 = []
inventory_level_2 = []

def observe(env):
    global reag_1
    while True:
        obs_time.append(env.now)
        inventory_level_1.append(reag_1)
        inventory_level_2.append(reag_2)
        yield env.timeout(0.1)

np.random.seed(0)
env = simpy.Environment()
env.process(coNet_run(env))
env.process(observe(env))
env.run(until=10.0)

import matplotlib.pyplot as plt
plt.figure()
plt.step(obs_time, inventory_level_1, where='post')
plt.xlabel('Simulation time (days)')
plt.ylabel('Fab1 Inventory level')
plt.figure()
plt.step(obs_time, inventory_level_2, where='post')
plt.xlabel('Simulation time (days)')
plt.ylabel('Fab2 Inventory level')
