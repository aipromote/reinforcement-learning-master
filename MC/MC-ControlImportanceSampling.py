#!/usr/bin/env python
# encoding: utf-8

import matplotlib
import numpy as np
import sys

from collections import defaultdict
from halften import HalftenEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

env = HalftenEnv()

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

xmajorLocator = MultipleLocator(0.5)
xmajorFormatter = FormatStrFormatter('%1.1f')

ymajorLocator = MultipleLocator(1)
ymajorFormatter = FormatStrFormatter('%d')

zmajorLocator = MultipleLocator(1)
zmajorFormatter = FormatStrFormatter('%d')

figureIndex = 0


def prettyPrint(data, tile, zlabel='reward'):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptitle(tile)
    fig.set_size_inches(18.5, 10.5)

    ax = fig.add_subplot(111, projection='3d')
    axisX = []
    axisY = []
    axisZ = []

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(1, 5)
    ax.set_zlim(0, 1)

    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)

    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)

    ax.zaxis.set_major_locator(zmajorLocator)
    ax.zaxis.set_major_formatter(zmajorFormatter)

    for i in data:
        axisX.append(i['x'])
        axisY.append(i['y'])
        axisZ.append(i['z'])
    ax.scatter(axisX, axisY, axisZ)
    ax.set_xlabel('player total score')
    ax.set_ylabel('player number of car')
    ax.set_zlabel(zlabel)


def create_random_policy(nA):
    '''
    Creates a random policy function.
    :param nA: Number of actions in the environment.
    :return: A function that takes an observation as input and returns a vector
            of action probabilities
    '''
    A = np.ones(nA, dtype=float) / nA

    def policy_fn(observation):
        return A

    return policy_fn


def create_greedy_policy(Q):
    '''
    Creates a greedy policy based on Q values.
    :param Q: A dictionary that maps from state -> action values
    :return: A function that takes an observation as input and returns a vector
        of action probabilities.
    '''

    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A

    return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    '''
     Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    :param env: Halften game
    :param num_episodes: Number of episodes to sample.
    :param behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
    :param discount_factor: Gamma discount factor.
    :return: A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    '''
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    target_policy = create_greedy_policy(Q)

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env._reset()
        for t in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env._step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        episode_len = len(episode)
        if episode_len < 4:
            continue

        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # Update the total reward since step t
            G = discount_factor * G + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * G - Q[state][action]
            # If the action taken by the behavior policy is not the action
            # taken by the target policy the probability will be 0 and we can break
            if action != np.argmax(target_policy(state)):
                break
            W = W * 1. / behavior_policy(state)[action]

    return Q, target_policy


random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)

policy_content = ["stop", "ask card"]

action_0_pcard = []
action_1_pcard = []
action_2_pcard = []
action_3_pcard = []
action_4_pcard = []

result = []

for state, actions in Q.items():
    action_value = np.max(actions)
    best_action = np.argmax(actions)
    score, card_num, p_num = state
    item = {"x": score, "y": int(card_num), "z": best_action, "p_num": p_num}
    result.append(item)

    if p_num == 0:
        action_0_pcard.append(item)
    elif p_num == 1:
        action_1_pcard.append(item)
    elif p_num == 2:
        action_2_pcard.append(item)
    elif p_num == 3:
        action_3_pcard.append(item)
    elif p_num == 4:
        action_4_pcard.append(item)

prettyPrint(action_0_pcard, "Optimal strategy when there is 0 card", "Use strategy")
prettyPrint(action_1_pcard, "Optimal strategy when there is 1 card", "Use strategy")
prettyPrint(action_2_pcard, "Optimal strategy when there is 2 card", "Use strategy")
prettyPrint(action_3_pcard, "Optimal strategy when there is 3 card", "Use strategy")
prettyPrint(action_4_pcard, "Optimal strategy when there is 4 card", "Use strategy")
plt.show()

# Output optimal strategies for each situation
result.sort(key=lambda obj: obj.get('x'), reverse=False)

for tmp in result:
    score = tmp["x"]
    card_num = tmp["y"]
    z = tmp["z"]
    p_num = tmp["p_num"]
    print(
        "The sum of the current number of hands is:%.1f,the current number of hands is:%d ,the current number of cards is:%d,the optimal strategy is:%s" % (
        score, card_num, p_num, policy_content[z]))
