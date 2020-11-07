import gym
import numpy as np


def compute_value_function(value_table, policy, gamma=1.0):
    threshold = 1e-8 # 设置置信区间

    while True:
        updated_value_table = np.copy(value_table)

        for state in range(env.nS):
            action = policy[state]
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in env.P[state][action]])

        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break

    return value_table


def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n) # 初始化新的策略

    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n) # 记录当前状态下，采取各个动作能收获的价值
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table) # 将价值最大的动作作为策略

    return policy


def policy_iteration(env, gamma=1.0):
    value_table = np.zeros(env.nS)  # 初始化值函数表
    old_policy = np.zeros(env.observation_space.n) # 初始化策略

    end_of_iterations = 200000 # 迭代终止条件

    for i in range(end_of_iterations):
        value_table = compute_value_function(value_table, old_policy, gamma) #更新值函数表
        new_policy = extract_policy(value_table, gamma) # 更新策略

        # 若策略已收敛，直接跳出循环
        if (old_policy == new_policy).all():
            print('Policy-Iteration converged at step %d.' % (i+1))
            break

        old_policy = new_policy

    return new_policy


env = gym.make('FrozenLake-v0')
env.render()
# print(env.P)

print(policy_iteration(env))
