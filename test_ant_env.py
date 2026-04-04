#!/usr/bin/env python3
import sys, time
sys.path.insert(0, '.')
import gymnasium as gym
import numpy as np

env = gym.make('Ant-v4')
obs, _ = env.reset(seed=0)
print('Ant-v4 obs dim:', env.observation_space.shape[0])
print('Ant-v4 act dim:', env.action_space.shape[0])
for _ in range(20):
    a = env.action_space.sample()
    obs, r, ter, trun, _ = env.step(a)
print('Ant-v4 OK, last reward:', round(r, 3))
env.close()

