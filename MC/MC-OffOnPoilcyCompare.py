#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys

from collections import defaultdict
from halften import HalftenEnv

env = HalftenEnv()


##################################On Policy##################################
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


def on_policy(state, Q, discount_factor, returns_sum, returns_count, episode, policy):
    '''
    On-Policy: Monte Carlo Control using Epsilon-Greedy policies.
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
        first_occurence_idx = next(i for i, x in enumerate(episode)
                                   if x[0] == state and x[1] == action)
        # Calculate the cumulative return from the first occurrence
        G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
        # Calculate the cumulative return mean of the current state
        returns_sum[sa_pair] += G
        returns_count[sa_pair] += 1.0
        # The improvement of strategy is to update the Q in this state.
        Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

    return Q, policy, returns_sum, returns_count, episode


##################################Off Policy##################################
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


def off_policy(state, behavior_policy, episode, discount_factor, target_policy, C, Q):
    '''
     Off-Policy: Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
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
    for t in range(100):
        probs = behavior_policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        next_state, reward, done, _ = env._step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

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

    return Q, target_policy, C, episode


def play(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    #####################On Policy Iterator Variable #####################
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q_on = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q_on, epsilon, env.action_space.n)

    #####################Off Policy Iterator Variable#####################
    Q_off = defaultdict(lambda: np.zeros(env.action_space.n))
    random_policy = create_random_policy(env.action_space.n)
    target_policy = create_greedy_policy(Q_off)

    C = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode_on = []
        episode_off = []

        state = env._reset()

        # On-Policy
        Q_on, policy, returns_sum, returns_count, episode_on = on_policy(state,
                                                                         Q_on,
                                                                         discount_factor,
                                                                         returns_sum,
                                                                         returns_count,
                                                                         episode_on,
                                                                         policy)

        # Off-Policy
        Q_off, target_policy, C, episode_off = off_policy(state,
                                                          random_policy,
                                                          episode_off,
                                                          discount_factor,
                                                          target_policy,
                                                          C,
                                                          Q_off)

    return Q_on, policy, Q_off, target_policy


Q_on, policy, Q_off, target_policy = play(env, num_episodes=500000, epsilon=0.1)
result_on = []
result_off = []

for state, actions in Q_on.items():
    action_value = np.max(actions)
    best_action = np.argmax(actions)
    score, card_num, p_num = state
    item = {"x": score, "y": int(card_num), "z": best_action, "p_num": p_num}
    result_on.append(item)

for state, actions in Q_off.items():
    action_value = np.max(actions)
    best_action = np.argmax(actions)
    score, card_num, p_num = state
    item = {"x": score, "y": int(card_num), "z": best_action, "p_num": p_num}
    result_off.append(item)

# sort
result_on.sort(key=lambda obj: obj.get('x'), reverse=False)
result_off.sort(key=lambda obj: obj.get('x'), reverse=False)

for on, off in zip(result_on, result_off):
    print(on, end="==>")
    print(off)
