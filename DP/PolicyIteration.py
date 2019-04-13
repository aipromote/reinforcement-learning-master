#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import PolicyEvaluation
from gridworld import GridworldEnv

# count operations times
v_num = 1
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


def policy_improvement(env, policy_eval_fn=PolicyEvaluation.policy_eval, discount_factor=1.0):
    '''
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    :param env: The OpenAI envrionment.
    :param policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
    :param discount_factor: gamma discount factor.
    :return: A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    '''
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        global i_num
        global v_num

        v_num = 1
        V = policy_eval_fn(policy, env, discount_factor)

        policy_stable = True

        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculated Q value
                    action_values[a] += prob * (reward + discount_factor * V[next_state])

            # Calculate all possibilities
            best_a_arr, policy_arr = get_max_index(action_values)

            if chosen_a not in best_a_arr:
                policy_stable = False
            policy[s] = policy_arr

        i_num = i_num + 1

        if policy_stable:
            print("After %d steps,the improvement done" % (i_num - 1))
            return policy, V


env = GridworldEnv()
policy, v = policy_improvement(env)

print("Draw Policy (0=up, 1=right, 2=down, 3=left):")
update_policy_type = change_policy(policy)
print(np.reshape(update_policy_type, env.shape))

print("Draw the improvement Value function:")
print(v.reshape(env.shape))