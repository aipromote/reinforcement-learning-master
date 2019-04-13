"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 50  # 如果全局回报总和大于此阈值则渲染环境
RENDER = False  # 是否渲染环境

env = gym.make('CartPole-v0')  # 加载环境CartPole(平衡杆)
env.seed(1)  # 设置随机种子,使结果可重现
env = env.unwrapped  # 取消限制

'''
env.action_space =  Discrete(2)
env.observation_space =  Box(4,)
env.observation_space.high =  [  4.80000019e+00   3.40282347e+38   4.18879032e-01   3.40282347e+38]
env.observation_space.low =  [ -4.80000019e+00  -3.40282347e+38  -4.18879032e-01  -3.40282347e+38]

n_actions= 2
n_features= 4
'''
print('env.action_space = ', env.action_space)
print('env.observation_space = ', env.observation_space)
print('env.observation_space.high = ', env.observation_space.high)
print('env.observation_space.low = ', env.observation_space.low)

print('n_actions=',env.action_space.n)
print('n_features=',env.observation_space.shape[0])

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    output_graph=True,
)

for i_episode in range(3000):
    # 环境的初始状态
    observation = env.reset()
    print('observation = ', observation)    # observation =  [ 0.03073904  0.00145001 -0.03088818 -0.03131252]

    while True:
        if RENDER: env.render()

        # 根据概率输出行为
        action = RL.choose_action(observation)
        # 根据行为获取下一次的状态,回报,是否终止和信息
        observation_, reward, done, info = env.step(action) # 因为环境的回报总是1.0,所以需要对回报做一些处理
        # print('action = ', action)
        # print('observation_ = ', observation_)
        # print('reward = ', reward)
        # print('done = ', done)
        # print('info = ', info)

        '''
        action =  0
        observation_ =  [ 0.03076804 -0.19321569 -0.03151444  0.25146705]
        reward =  1.0
        done =  False
        info =  {} 
       '''
        # 将该次的结果存入记忆库中
        RL.store_transition(observation, action, reward)

        if done:
            # 回合结束后开始进行学习
            print('RL.ep_rs = ', RL.ep_rs)
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # 进行环境的渲染
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)  # 展示每回合的vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
