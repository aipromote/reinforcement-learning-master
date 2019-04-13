#!/usr/bin/env python
# encoding: utf-8

from maze import Maze
from RL_brain import SarsaLambdaTable, QLambdaTable
import numpy as np

METHOD = "QLambda"


def get_action(q_table, state):
    state_action = q_table.ix[state, :]
    state_action_max = state_action.max()

    idxs = []
    for max_item in range(len(state_action)):
        if state_action[max_item] == state_action_max:
            idxs.append(max_item)

    sorted(idxs)
    return tuple(idxs)


def get_policy(q_table, rows=6, cols=6, pixels=40, orign=20):
    policy = []

    for i in range(rows):
        for j in range(cols):
            item_center_x, item_center_y = (j * pixels + orign), (i * pixels + orign)
            item_state = [item_center_x - 15.0, item_center_y - 15.0, item_center_x + 15.0, item_center_y + 15.0]

            # If the current state is each terminated state, the value is -1
            if item_state in [env.canvas.coords(env.hell1), env.canvas.coords(env.hell2),
                              env.canvas.coords(env.hell3), env.canvas.coords(env.hell4), env.canvas.coords(env.oval)]:
                policy.append(-1)
                continue

            if str(item_state) not in q_table.index:
                policy.append((0, 1, 2, 3))
                continue

            item_action_max = get_action(q_table, str(item_state))
            policy.append(item_action_max)

    return policy


def judge(observation):
    '''
    Determine whether the current state is in the secondary air duct
    :param observation: current state
    :return:
    '''
    x = (observation[0] + observation[2]) / 2

    # When the x is 140, it is a duct
    if x == 140:
        return True
    return False


def update():
    for episode in range(1000):
        observation = env.reset()

        # Select behavior based on current state
        action = RL.choose_action(str(observation))
        # Initialize all eligibility_trace to 0
        RL.eligibility_trace *= 0

        while True:
            env.render()

            # In game,he position of the secondary wind will go up two squares,
            # Determine whether the current state is in the secondary air duct and the generated action is an upward motion
            if judge(observation) and action == 0:
                observation_, reward, done, oval_flag = env.step(action)

                # If the termination state occurs during the process, it ends directly
                if done:
                    break

                # Direct assignment is continued upwards, and reward add
                action_ = 0
                reward = 0.1
                RL.learn(str(observation), action, reward, str(observation_), action_)
                observation = observation_
                action = action_

            # Take action from the current state to get the observation_, reward, done, oval_flag
            observation_, reward, done, oval_flag = env.step(action)
            # Based on the next state selection behavior
            action_ = RL.choose_action(str(observation_))

            # If you go down the wind tunnel, you will do special treatment when you are not in the trap (to prevent the return of the wind tunnel to increase the reward)
            if judge(observation) and action == 1:
                reward = -0.1

            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_

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
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    if METHOD == "QLambda":
        RL = QLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
