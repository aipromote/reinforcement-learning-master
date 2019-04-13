#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lib.envs.maze import Maze
from RL_brain import QLearningTable, EnvModel
import matplotlib.pyplot as plt

import numpy as np


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


def show_plot(result):
    plt.figure(figsize=(15, 4))

    for planning_step in result:
        plt.plot(result[planning_step].keys(), result[planning_step].values(),
                 label='%s planning steps' % planning_step)

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.title('DynaQ Algorithm Maze')

    plt.show()


def update():
    planning_steps = [0, 5, 50]

    for planning_step in planning_steps:
        result[planning_step] = {}
        # clear q table
        RL.clear_q_table()
        # Empty the memory in the simulation environment
        env_model.clear_database()

        for episode in range(50):
            s = env.reset()
            step = 0

            while True:
                env.render()
                a = RL.choose_action(str(s))
                s_, r, done, oval_flag = env.step(a)
                RL.learn(str(s), a, r, str(s_))

                env_model.store_transition(str(s), a, r, s_)
                # Use env_model to learn planning_step times
                for n in range(planning_step):
                    ms, ma = env_model.sample_s_a()
                    mr, ms_ = env_model.get_r_s_(ms, ma)
                    RL.learn(ms, ma, mr, str(ms_))

                s = s_

                step += 1
                if done:
                    result[planning_step][episode] = step
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
    show_plot(result)


if __name__ == "__main__":
    # 2018-10-25需求添加: 绘制规划次数n和每回合收敛步数的关系图
    result = {}

    # 创建迷宫环境
    env = Maze()

    print(env.n_actions)
    print(list(range(env.n_actions)))
    # 创建Q-Learning决策对象
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # 创建环境模型
    env_model = EnvModel(actions=list(range(env.n_actions)))

    env.after(0, update)
    env.mainloop()
