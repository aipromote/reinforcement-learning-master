#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # If the status does not exist in the current Q table, add the current status to the Q table.
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # Randomly sampled from the uniformly distributed [0,1). When less than the threshold, the method of selecting the optimal behavior is adopted. When the random behavior is selected above the threshold, the artificial randomness is added to solve the local optimality.
        if np.random.rand() < self.epsilon:
            state_action = self.q_table.ix[observation, :]
            # Because there may be more than one optimal behavior in a state, when you encounter this situation, you need to randomly select a behavior.
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def judge(self, observation, action_):
        self.check_state_exist(observation)
        state_action = self.q_table.ix[observation, :]
        max_num = state_action.max()
        idxs = []

        for max_item in range(len(state_action)):
            if state_action[max_item] == max_num:
                idxs.append(max_item)
        if action_ in idxs:
            return True
        return False

    def learn(self, *args):
        pass


class SarsaLambdaTable(RL):
    '''
    Sarsa(λ)
    '''

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Add a new state to the Q table
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)
            # Update eligibility_trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_, ):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict
        self.eligibility_trace.ix[s, a] += 1

        # Update Q table
        self.q_table += self.lr * error * self.eligibility_trace
        # Update eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_


class QLambdaTable(RL):
    '''
    Q(λ)--Watkins's Q(λ)
    '''

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(QLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Add a new state to the Q table
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)
            # Update eligibility_trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_].max()
        else:
            q_target = r
        error = q_target - q_predict
        a_flag = self.judge(s, a_)
        self.eligibility_trace.ix[s, a] += 1

        # Update Q table
        self.q_table += self.lr * error * self.eligibility_trace
        # Update eligibility_trace
        if a_flag:
            self.eligibility_trace *= self.gamma * self.lambda_
        else:
            self.eligibility_trace *= 0
