#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import gym
from RL_brain import DDPG

MAX_EPISODES = 200
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000

RENDER = False
ENV_NAME = 'Pendulum-v0'

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Increase noise during exploration
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        # Store the (s, a, r, s_) in the memory
        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300: RENDER = True
            break
