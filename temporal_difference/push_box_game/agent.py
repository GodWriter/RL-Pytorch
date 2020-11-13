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
            state_action = state_action.reindex(np.random.permutation(state_action.index)) # 重新排序原索引对应的值，避免价值相同的动作被重复选择
            action_idx = state_action.idxmax() # 索引及对应值重新排序后，获取最大值所对应的索引
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
    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)

        q_predict = self.q_table[s][a]
        if not done:
            q_target = r + self.reward_decay * self.q_table[s_].max() # Q-learning根据贪婪策略选择动作
        else:
            q_target = r # 如果是终态，该收益为最终收益
        self.q_table[s][a] += self.learning_rate * (q_target - q_predict)

    def train(self, env, max_iterator=100):
        self.load_train_parameter("q_table.pkl")

        for episode in range(max_iterator):
            observation = env.reset()

            while True:
                env.render()

                action_idx = self.choose_action(observation)
                observation_, reward, done, _ = env.step(self.actions[action_idx])

                self.learn(observation, action_idx, reward, observation_, done)
                observation = observation_

                if done: # 要么推箱子推到墙壁了，要么推到终点了
                    self.save_train_parameter("q_table.pkl")
                    break


class SarsaAgent(Agent):
    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)

        q_predict = self.q_table[s][a]
        if not done:
            q_target = r + self.reward_decay * self.q_table[s_][self.choose_action(s_)] # Sarsa根据依旧根据 epsilon-贪婪策略 选择动作
        else:
            q_target = r # 如果是终态，该收益为最终收益
        self.q_table[s][a] += self.learning_rate * (q_target - q_predict)

    def train(self, env, max_iterator=100):
        self.load_train_parameter("q_table.pkl")

        for episode in range(max_iterator):
            observation = env.reset()

            while True:
                env.render()

                action_idx = self.choose_action(observation)
                observation_, reward, done, _ = env.step(self.actions[action_idx])

                self.learn(observation, action_idx, reward, observation_, done)
                observation = observation_

                if done: # 要么推箱子推到墙壁了，要么推到终点了
                    self.save_train_parameter("q_table.pkl")
                    break
