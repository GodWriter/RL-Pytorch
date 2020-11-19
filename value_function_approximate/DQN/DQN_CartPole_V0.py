import gym
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        action_value = self.fc2(x)
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


def main(args):
    agent = DQN(args)

    for n_ep in range(args.n_episodes):
        state = env.reset()
        if args.render: env.render()

        for t in range(10000):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            if args.render: env.render()
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)

            if done or t >= 9999:
                agent.writer.add_scalar('DQN_CartPole_V0/finish_step', t+1, global_step=n_ep) # 记录每一轮在第几步结束
                agent.learn()

                if n_ep % 10 == 0: print("episode {}, step is {}".format(n_ep, t))
                break

    print("Training is Done!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_episodes', type=int, default=2000)
    parser.add_argument('--capacity', type=int, default=8000)
    parser.add_argument('--update_count', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--logs', type=str, default='logs/DQN_CartPole_V0')
    args = parser.parse_args()
    print(args)

    main(args)
