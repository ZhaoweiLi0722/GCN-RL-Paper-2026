#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 00:21:18 2020

@author: howard
"""

import gym
import gym_foo_v1
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent0 import Agent

env = gym.make('foo-v1')
env.seed(0)
print('State shape: ', env.observation_space.shape) #State shape:  (14,)
print('Number of actions: ', env.action_space.shape) #Number of actions:  (5,)
agent = Agent(state_size=19, action_size=5, random_seed=1)

def ddpg(n_episodes=10000, max_t=200, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _, next_state_true = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
