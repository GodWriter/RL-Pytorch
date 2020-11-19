import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from collections import namedtuple
from tensorboardX import SummaryWriter

import time

import numpy as np
import tkinter as tk


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
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x) # 输出为动作所对应的得分
        return action_value


class DQN():
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args

        self.target_net = Net()
        self.eval_net = Net()
        self.memory = [None] * self.args.capacity

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.args.learning_rate)

        self.memory_count = 0
        self.update_count = 0
        self.writer = SummaryWriter(self.args.logs)

    def store_transition(self, transition):
        idx = self.memory_count % self.args.capacity

        self.memory[idx] = transition
        self.memory_count += 1

        return self.memory_count >= self.args.capacity

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.eval_net(state)

        _, idx = torch.max(value, 1)
        action = idx.item() # Important, get the action idx outside the graph

        if np.random.rand(1) >= 0.9: # epsilon greedy
            action = np.random.choice(range(num_action), 1).item()

        return action

    def learn(self):
        if self.memory_count >= self.args.capacity:
            state = torch.tensor([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1, 1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.tensor([t.next_state for t in self.memory]).float()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad(): # 对target_v计算中所有涉及到的图节点不进行更新
                target_v = reward + self.args.gamma * self.target_net(next_state).max(1)[0]

            # Training
            for idx in BatchSampler(SubsetRandomSampler(range(len(self.memory))),
                                    batch_size=self.args.batch_size,
                                    drop_last=False):
                eval_v = (self.eval_net(state).gather(1, action))[idx]
                loss = self.loss_func(target_v[idx].unsqueeze(1), eval_v)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count += 1

                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.eval_net.state_dict())
        else:
            print("Memory Buff size is {}, Now is collecting more...".format(self.memory_count))


def main(args):
    agent = DQN(args)

    for n_ep in range(args.n_episodes):
        state = env.reset()
        ep_reward = 0.0 # 记录一个episode能够拿到的整体收益
        if args.render: env.render()

        for t in range(10000):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward

            if args.render: env.render()
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)

            if done or t >= 9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=n_ep) # 记录每一轮在第几步结束
                agent.learn()

                if n_ep % 10 == 0:
                    print("episode {}, step is {}, the episode reward is {}".format(n_ep, t, round(ep_reward, 3)))
                break

            state = next_state

    print("Training is Done!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_episodes', type=int, default=2000)
    parser.add_argument('--capacity', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--logs', type=str, default='logs/DQN_PushBox_Game')
    args = parser.parse_args()
    print(args)

    main(args)
