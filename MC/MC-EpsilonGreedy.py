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


def make_epsilon_greedy_policy(Q, epsilon, nA):
    '''
     Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    :param Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
    :param epsilon: The probability to select a random action . float between 0 and 1.
    :param nA: Number of actions in the environment.
    :return: A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
    '''

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    '''
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    :param env: Halften games
    :param num_episodes: Number of episodes to sample.
    :param discount_factor: Gamma discount factor.
    :param epsilon: Chance the sample a random action. Float betwen 0 and 1.
    :return: A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns
            action probabilities
    '''
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env._reset()

        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env._step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Calculate the cumulative return from the first occurrence
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            # The improvement of strategy is to update the Q in this state.
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    return Q, policy


Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
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
result.sort(key=lambda obj:obj.get('x'), reverse=False)

for tmp in result:
    score = tmp["x"]
    card_num = tmp["y"]
    z = tmp["z"]
    p_num = tmp["p_num"]
    print("The sum of the current number of hands is:%.1f,the current number of hands is:%d ,the current number of cards is:%d,the optimal strategy is:%s" % (score, card_num, p_num, policy_content[z]))
