import gym
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from collections import namedtuple
from tensorboardX import SummaryWriter


# Environment Config
env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

seed = 1
env.seed(seed)
torch.manual_seed(seed)


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
            print("Memory Buff is too less, Now is collecting...")


def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

    return r1 + r2 # 需自己定义reward，gym提供的会使得模型无法收敛


def main(args):
    agent = DQN(args)

    for n_ep in range(args.n_episodes):
        state = env.reset()
        ep_reward = 0.0 # 记录一个episode能够拿到的整体收益
        if args.render: env.render()

        for t in range(10000):
            action = agent.choose_action(state)
            next_state, _, done, info = env.step(action)

            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)
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
    parser.add_argument('--capacity', type=int, default=8000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--logs', type=str, default='logs/DQN_CartPole_V0')
    args = parser.parse_args()
    print(args)

    main(args)
