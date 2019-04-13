#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Bandits(object):

    def __init__(self, probs, rewards):
        '''
        :param probs: float[],the probability of winning the Bandits
        :param rewards: float[],reward value when winning
        '''

        if len(probs) != len(rewards):
            raise Exception('The winning probability array does not match the reward array length!')
        self.probs = probs
        self.rewards = rewards

    def pull(self, i):
        '''
        The reward of i# bandit
        :param i: int, Number of Bandit
        :return: float or None
        '''

        if np.random.rand() < self.probs[i]:
            return self.rewards[i]
        else:
            return 0.0


class Algorithm(object):

    def __init__(self, operate):
        self.operate = operate

    def eps_greedy(self, params):
        '''
        ε-greedy
        :param params: ε
        :return: Number of Bandit
        '''
        if params and type(params) == dict:
            eps = params.get('epsilon')
        else:
            eps = 0.1

        r = np.random.rand()

        if r < eps:
            return np.random.choice(
                list(set(range(len(self.operate.wins))) - {np.argmax(self.operate.wins / (self.operate.pulls + 0.1))}))
        else:
            return np.argmax(self.operate.wins / (self.operate.pulls + 0.1))

    def ucb(self, params=None):
        '''
        UCB1
        :param params: None
        :return: Number of Bandit
        '''
        if True in (self.operate.pulls < self.operate.num_bandits):
            return np.random.choice(range(len(self.operate.pulls)))
        else:
            n_tot = sum(self.operate.pulls)
            rewards = self.operate.wins / (self.operate.pulls + 0.1)
            ubcs = rewards + np.sqrt(2 * np.log(n_tot) / self.operate.pulls)

            return np.argmax(ubcs)

    def ts(self, params=None):
        '''
        Thompson Sampling
        :param params: None
        :return: Number of Bandit
        '''
        p_success_arms = [
            np.random.beta(self.operate.wins[i] + 1, self.operate.pulls[i] - self.operate.wins[i] + 1)
            for i in range(len(self.operate.wins))
        ]

        return np.array(p_success_arms).argmax()


class Operate(object):

    def __init__(self, num_bandits=10, probs=None, rewards=None, strategies=['eps_greedy', 'ucb', 'ts']):
        '''
        :param num_bandits: int, Number of Bandits (default: 10)
        :param probs: float[], Probability of winning
        :param rewards: float[], Reward value after winning
        :param strategies: str[], Policy value
        '''
        self.choices = []

        if not probs:
            if not rewards:
                self.bandits = Bandits(probs=[np.random.rand() for idx in range(num_bandits)],
                                       rewards=np.ones(num_bandits))
            else:
                self.bandits = Bandits(probs=[np.random.rand() for idx in range(len(rewards))], rewards=rewards)
                num_bandits = len(rewards)
        else:
            if rewards:
                self.bandits = Bandits(probs=probs, rewards=rewards)
                num_bandits = len(rewards)
            else:
                self.bandits = Bandits(probs=probs, rewards=np.ones(len(probs)))
                num_bandits = len(probs)

        self.num_bandits = num_bandits
        self.wins = np.zeros(num_bandits)
        self.pulls = np.zeros(num_bandits)

        # The optional strategy of the Bandit, defaults to all strategies in the algorithm class
        self.strategies = strategies
        self.algorithm = Algorithm(self)

    def run(self, time=100, strategy='eps_greedy', parameters={'epsilon': 0.1}):
        '''
        Operate Run times
        :param time: int, Run times
        :param strategy: str, Policy name, default is ε-greedy strategy
        :param parameters: dict, Policy parameters, default ε=0.1
        :return: None
        '''
        if int(time) < 1:
            raise Exception('Times should be greater than 1!')

        if strategy not in self.strategies:
            raise Exception(
                'The incoming policy is not supported, please select a policy: {}'.format(', '.join(self.strategies)))

        for n in range(time):
            self._run(strategy, parameters)

    def _run(self, strategy, parameters=None):
        '''
        Incoming policy in a single run operation
        :param strategy: str, Policy name
        :param parameters: dict, Policy parameters
        :return: None
        '''

        choice = self.run_strategy(strategy, parameters)
        self.choices.append(choice)
        rewards = self.bandits.pull(choice)
        if rewards is None:
            return None
        else:
            self.wins[choice] += rewards
        self.pulls[choice] += 1

    def run_strategy(self, strategy, parameters):
        '''
        Run the strategy and return the behavior of the Bandit selection
        :param strategy: str, Policy name
        :param parameters: dict, Policy parameters
        :return: Number of Bandit
        '''

        return self.algorithm.__getattribute__(strategy)(params=parameters)

    def regret(self):
        '''
        Calculate the regret value
        Expected regret = maximum reward - the sum of the rewards collected,
        example:  E(target)= T * max_k（mean_k） - sum_（t = 1 - > T）（reward_t）
        :return:
        '''

        return (sum(self.pulls) * np.max(np.nan_to_num(self.wins / (self.pulls + 0.1))) - sum(self.wins)) / (
                sum(self.pulls) + 0.1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    # All strategy
    strategies = [{'strategy': 'eps_greedy', 'regret': [],
                   'label': '$\epsilon$-greedy ($\epsilon$=0.1)'},
                  {'strategy': 'ucb', 'regret': [],
                   'label': 'UCB1'},
                  {'strategy': 'ts', 'regret': [],
                   'label': 'Thompson Sampling'}
                  ]
    for s in strategies:
        s['mab'] = Operate()

    for t in range(1000):
        for s in strategies:
            s['mab'].run(strategy=s['strategy'])
            s['regret'].append(s['mab'].regret())

    sns.set_style('whitegrid')
    sns.set_context('poster')

    plt.figure(figsize=(15, 4))
    for s in strategies:
        plt.plot(s['regret'], label=s['label'])

    plt.legend()
    plt.xlabel('Trials')
    plt.ylabel('Regret')
    plt.title('Multi-armed bandit strategy performance')

    plt.show()
