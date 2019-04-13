#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from gridworld import GridworldEnv


def policy_eval(policy, env, discount_factor=1):
    '''
    This method is used as a policy evaluation
    :param policy: Pending policy
    :param env: The OpenAI envrionment.
    :param discount_factor: gamma discount factor.
    :return: V is the value function for the optimal policy.
    '''
    V = np.zeros(env.nS)
    i = 0

    while True:
        value_delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    # Solving the state value function using the Bellman expectation equation
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            value_delta = max(value_delta, np.abs(v - V[s]))
            V[s] = v
        i += 1
        # If the maximum difference between each state and the previous state obtained by the current loop is less than the threshold, the convergence stops.
        if value_delta == 0:
            print("After %d steps,the eval done" % i)
            break
    return np.array(V)

if __name__ == '__main__':
    # testing
    env = GridworldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(random_policy, env)

    print("Draw the eval Value function:")
    print(v.reshape(env.shape))
