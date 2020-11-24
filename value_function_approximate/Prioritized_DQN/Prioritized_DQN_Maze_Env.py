import math
import time
import argparse

import numpy as np
import tkinter as tk

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter


seed = 1
torch.manual_seed(seed)


UNIT = 40   # 像素单位长度
MAZE_H = 4  # 高度
MAZE_W = 4  # 宽度


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()

        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)

        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 创建格子
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 创建 origin
        origin = np.array([20, 20])

        # hell1
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(hell1_center[0] - 15,
                                                  hell1_center[1] - 15,
                                                  hell1_center[0] + 15,
                                                  hell1_center[1] + 15,
                                                  fill='black')

        # hell2
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(hell2_center[0] - 15,
                                                  hell2_center[1] - 15,
                                                  hell2_center[0] + 15,
                                                  hell2_center[1] + 15,
                                                  fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(oval_center[0] - 15,
                                            oval_center[1] - 15,
                                            oval_center[0] + 15,
                                            oval_center[1] + 15,
                                            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(origin[0] - 15,
                                                 origin[1] - 15,
                                                 origin[0] + 15,
                                                 origin[1] + 15,
                                                 fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)

        self.canvas.delete(self.rect)

        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0] - 15,
                                                 origin[1] - 15,
                                                 origin[0] + 15,
                                                 origin[1] + 15,
                                                 fill='red')

        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect) # 返回的是中心点坐标，所以下面比较的都是中心点的位置

        base_action = np.array([0, 0])
        if action == 0: # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1: # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2: # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1]) # 移动智能体
        s_ = self.canvas.coords(self.rect) # 得到下一状态

        # reward function
        if s_ == self.canvas.coords(self.oval): # 走出迷宫
            reward = 1
            done = True
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]: # 走入陷阱
            reward = -1
            done = True
        else: # 其他位置
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


# Environment Config
env = Maze()

num_state = 4
num_action = env.n_actions

# CUDA
device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
device = torch.device(device)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.capacity = capacity
        self.prob_alpha = prob_alpha

        self.pos = 0
        self.buffer = []

        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert len(state) == len(next_state)

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0 # 初始的概率都是等高的，这个设置1，2，3都可以，相等即可

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio # 令新样本优先级为当前最高，以保证每个样本至少被利用一次
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # 根据概率分布采样一个样本点
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 计算样本点的重要性权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = zip(*samples)
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio # TD偏差越大的，被采样到的概率越高

    def __len__(self):
        return len(self.buffer)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(num_state, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_action))

    def forward(self, x):
        return self.layers(x)


class PrioritizedDQN():
    def __init__(self, args):
        super(PrioritizedDQN, self).__init__()
        self.args = args

        self.target_net = Net().to(device)
        self.eval_net = Net().to(device)

        self.memory_count = 0
        self.update_count = 0
        self.replay_buffer = NaivePrioritizedBuffer(self.args.capacity)

        self.writer = SummaryWriter(self.args.logs)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.args.learning_rate)

    def choose_action(self, state, epsilon):
        if np.random.rand(1) > epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            value = self.eval_net(state)

            _, idx = torch.max(value, 1)
            action = idx.item()
        else:
            action = np.random.choice(range(num_action), 1).item()

        return action

    def learn(self, beta):
        if self.memory_count >= self.args.capacity:
            state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.args.batch_size, beta)

            state = torch.FloatTensor(np.float32(state)).to(device)
            next_state = torch.FloatTensor(np.float32(next_state)).to(device)
            action = torch.LongTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
            weights = torch.FloatTensor(weights).to(device)

            eval_v = self.eval_net(state)
            target_v = self.target_net(next_state)

            q_v = eval_v.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_v = target_v.max(1)[0]
            expected_q_v = reward + self.args.gamma * next_q_v * (1 - done)

            loss = (q_v - expected_q_v.detach()).pow(2) * weights
            prios = loss + 1e-5
            loss = loss.mean()
            self.replay_buffer.update_priorities(indices, prios.detach().cpu().numpy())  # TD偏差用于更新样本的优先级

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('loss/value_loss', loss, self.update_count)
            self.update_count += 1

            if self.update_count % 100 == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())


def epsilon_by_frame(frame_idx):
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


def beta_by_frame(frame_idx):
    beta_start = 0.4
    beta_frames = 1000

    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


def reward_func(env, x, theta):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

    return r1 + r2 # 需自己定义reward，gym提供的会使得模型无法收敛


def main(args):
    agent = PrioritizedDQN(args)

    ep_reward = 0.0
    state = env.reset()
    if args.render: env.render()

    for f_idx in range(args.n_frames):
        # choose action and get next_state
        epsilon = epsilon_by_frame(f_idx)
        action = agent.choose_action(state, epsilon)
        next_state, reward, done = env.step(action)

        # update the replay buffer.
        if args.render: env.render()
        agent.replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        ep_reward += reward

        if len(agent.replay_buffer) > args.batch_size:
            beta = beta_by_frame(f_idx)
            agent.learn(beta)

        if f_idx % 200 == 0:
            print("Frame is {}, the episode reward is {}".format(f_idx, round(ep_reward, 3)))

        if done:
            state = env.reset()
            ep_reward = 0.0

    print("Training is Done!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_frames', type=int, default=1000000)
    parser.add_argument('--capacity', type=int, default=8000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--logs', type=str, default='logs/Prioritized_DQN_CartPole_V0')
    args = parser.parse_args()
    print(args)

    main(args)
