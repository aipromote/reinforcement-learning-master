#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import random
from collections import deque

FRAME_PER_ACTION = 1
GAMMA = 0.99
OBSERVE = 100.
EXPLORE = 200000.
FINAL_EPSILON = 0.001
INITIAL_EPSILON = 0.01
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
UPDATE_TIME = 100

try:
    tf.mul
except:
    tf.mul = tf.multiply


class BrainDQN:

    def __init__(self, actions):
        self.replayMemory = deque()
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON

        self.saved = 0
        self.actions = actions
        init_params = self.createQNetwork()
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = init_params

        target_params = self.createQNetwork()
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = target_params

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_path = checkpoint.model_checkpoint_path

            self.saver.restore(self.session, checkpoint_path)
            print("Successfully loaded:", checkpoint_path)

            self.saved = int(checkpoint_path.split('-')[-1])
        else:
            print("Could not find old network weights")

    def createQNetwork(self):
        '''
        使用网络为卷积神经网络
        卷积神经网络（CNN）由输入层、卷积层、激活函数、池化层、全连接层组成
        卷积层：用它来进行特征提取
        池化层: 池化操作可以减小数据量，从而减小参数，降低计算，因此防止过拟合
        输入图像(80x80x4) --> 卷积(卷积核(8x8x4x32),步长4) (20x20x32) --> 池化(池化核(2x2)) (10x10x32) -->
                          --> 卷积(卷积核(4x4x32x64),步长2) (5x5x64) --> 池化(池化核(2x2)) (3x3x64) -->
                          --> 卷积(卷积核(3x3x64x64),步长1) (3x3x64) --> 池化(池化核(2x2)) (2x2x64) -->
                          --> 变形(256x1) --> 全连接并使用RELU激活(512x1) --> 数组相乘(2x1)
        '''
        # Network weight
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([256, 256])
        b_fc1 = self.bias_variable([256])

        W_fc2 = self.weight_variable([256, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # Input
        stateInput = tf.placeholder("float", [None, 80, 80, 4])

        # Hidden layer
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 1) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        h_conv3_flat = tf.reshape(h_pool3, [-1, 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # 求出Q值
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def trainQNetwork(self):
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # Save the current training node every 10,000 iterations
        if (self.timeStep + self.saved) % 1000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn',
                            global_step=(self.saved + self.timeStep))

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            self.trainQNetwork()

        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", (self.saved + self.timeStep), "/ STATE", state, \
              "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1

        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
