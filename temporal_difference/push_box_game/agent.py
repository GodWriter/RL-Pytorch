import random
import os

import numpy as np
import pandas as pd


class Agent(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.epsilon = e_greedy

        self.reward_decay = reward_decay
        self.learning_rate = learning_rate

        self.q_table = pd.DataFrame()

    def check_state_exist(self, state):
        if tuple(state) not in self.q_table.columns:
            self.q_table[tuple(state)] = [0] * len(self.actions)

    def choose_action(self, observation):
        observation = tuple(observation)
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon: # epsilon-greedy policy
            state_action = self.q_table[observation]
            state_action = state_action.reindex(np.random.permutation(state_action.index)) # 避免出现相同值，导致重复选择同一个动作
            action_idx = state_action.argmax()
        else:
            action_idx = np.random.choice(range(len(self.actions)))

        return action_idx

    def save_train_parameter(self, name):
        self.q_table.to_pickle(name)

    def load_train_parameter(self, name):
        self.q_table = pd.read_pickle(name)

    def learn(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass


class QLearningAgent(Agent):
    pass


class SarsaAgent(Agent):
    pass
