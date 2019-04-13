#!/usr/bin/env python
# encoding: utf-8

from maze import Maze
from RL_brain import QLearningTable, SarsaTable
import numpy as np

METHOD = "Q-Learning"


def get_action(q_table, state):
    state_action = q_table.ix[state, :]
    state_action_max = state_action.max()

    idxs = []
    for max_item in range(len(state_action)):
        if state_action[max_item] == state_action_max:
            idxs.append(max_item)

    sorted(idxs)
    return tuple(idxs)


def get_policy(q_table, rows=5, cols=5, pixels=40, orign=20):
    policy = []

    for i in range(rows):
        for j in range(cols):
            item_center_x, item_center_y = (j * pixels + orign), (i * pixels + orign)
            item_state = [item_center_x - 15.0, item_center_y - 15.0, item_center_x + 15.0, item_center_y + 15.0]

            # If the current state is each terminated state, the value is -1
            if item_state in [env.canvas.coords(env.hell1), env.canvas.coords(env.hell2),
                              env.canvas.coords(env.hell3), env.canvas.coords(env.hell4),
                              env.canvas.coords(env.hell5), env.canvas.coords(env.hell6),
                              env.canvas.coords(env.hell7), env.canvas.coords(env.oval)]:
                policy.append(-1)
                continue

            if str(item_state) not in q_table.index:
                policy.append((0, 1, 2, 3))
                continue
            item_action_max = get_action(q_table, str(item_state))
            policy.append(item_action_max)

    return policy


def update():
    for episode in range(100):
        observation = env.reset()

        c = 0
        tmp_policy = {}

        while True:
            env.render()
            # Select behavior based on current state
            action = RL.choose_action(str(observation))
            state_item = tuple(observation)
            tmp_policy[state_item] = action
            # Take action to get the next state and return, and whether to terminate
            observation_, reward, done, oval_flag = env.step(action)

            if METHOD == "SARSA":
                # Based on the next state selection behavior
                action_ = RL.choose_action(str(observation_))
                # Update Q based on (s, a, r, s_, a_) using Sarsa
                RL.learn(str(observation), action, reward, str(observation_), action_)
            elif METHOD == "Q-Learning":
                # Update Q using Q-Learning
                RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            c += 1
            if done:
                break
    print('Game Over')
    q_table_result = RL.q_table
    policy = get_policy(q_table_result)
    print("The optimal strategy is", end=":")
    print(policy)

    print("Draw Policy", end=":")
    policy_result = np.array(policy).reshape(5, 5)
    print(policy_result)

    print("Drawing path: ")
    env.render_by_policy(policy_result)


if __name__ == "__main__":
    env = Maze()

    RL = SarsaTable(actions=list(range(env.n_actions)))

    if METHOD == "Q-Learning":
        RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
