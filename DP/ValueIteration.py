#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from gridworld import GridworldEnv

# count operations times
i_num = 1


def get_max_index(action_values):
    indexs = []
    policy_arr = np.zeros(len(action_values))

    action_max_value = np.max(action_values)

    for i in range(len(action_values)):
        action_value = action_values[i]

        if action_value == action_max_value:
            indexs.append(i)
            policy_arr[i] = 1
    return indexs, policy_arr


def change_policy(policys):
    action_tuple = []

    for policy in policys:
        indexs, policy_arr = get_max_index(policy)
        action_tuple.append(tuple(indexs))

    return action_tuple


def value_iteration(env, threshold=0, discount_factor=1.0):
    '''
    Value Iteration Algorithm.
    :param env: The OpenAI envrionment.
    :param threshold: Convergence condition
    :param discount_factor: gamma discount factor.
    :return: A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    '''
    global i_num

    def one_step_lookahead(state, V):
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                q[a] += prob * (reward + discount_factor * V[next_state])
        return q

    V = np.zeros(env.nS)

    while True:
        delta = 0
        for s in range(env.nS):
            q = one_step_lookahead(s, V)
            best_action_value = np.max(q)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value

        i_num += 1
        if delta == threshold:
            print("After %d steps,The state's optimal behavior value function has converged and the operation ends." % (
                    i_num - 1))
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        q = one_step_lookahead(s, V)
        best_a_arr, policy_arr = get_max_index(q)
        policy[s] = policy_arr
    return policy, V


env = GridworldEnv()
policy, v = value_iteration(env)

print("Draw Policy (0=up, 1=right, 2=down, 3=left):")
update_policy_type = change_policy(policy)
print(np.reshape(update_policy_type, env.shape))

print("Draw the improvement Value function:")
print(v.reshape(env.shape))
