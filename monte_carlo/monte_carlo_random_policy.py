import random


class GridMdp:
    def __init__(self):
        self.gamma = 0.9
        self.states = range(1, 26) # 状态空间

        self.actions = ['n', 'e', 's', 'w'] # 动作空间
        self.terminate_states = {15: 1.0, 4: -1.0, 9: -1.0, 11: -1.0, 12: -1.0, 23: -1.0, 24: -1.0, 25: -1.0}

        self.trans = {} # 概率转移矩阵
        for state in self.states:
            if not state in self.terminate_states:
                self.trans[state] = {}

        self.trans[1]['e'] = 2
        self.trans[1]['s'] = 6
        self.trans[2]['e'] = 3
        self.trans[2]['w'] = 1
        self.trans[2]['s'] = 7
        self.trans[2]['s'] = 7
        self.trans[3]['e'] = 4
        self.trans[3]['w'] = 2
        self.trans[3]['s'] = 8
        self.trans[5]['s'] = 10
        self.trans[6]['e'] = 7
        self.trans[6]['s'] = 11
        self.trans[6]['n'] = 1
        self.trans[7]['e'] = 8
        self.trans[7]['w'] = 6
        self.trans[7]['s'] = 12
        self.trans[7]['n'] = 2
        self.trans[8]['e'] = 9
        self.trans[8]['w'] = 7
        self.trans[8]['s'] = 13
        self.trans[8]['n'] = 3
        self.trans[10]['w'] = 9
        self.trans[10]['s'] = 15
        self.trans[13]['e'] = 14
        self.trans[13]['w'] = 12
        self.trans[13]['s'] = 18
        self.trans[13]['n'] = 8
        self.trans[14]['e'] = 15
        self.trans[14]['w'] = 13
        self.trans[14]['s'] = 19
        self.trans[14]['n'] = 9
        self.trans[16]['e'] = 17
        self.trans[16]['s'] = 21
        self.trans[16]['n'] = 11
        self.trans[17]['e'] = 18
        self.trans[17]['w'] = 16
        self.trans[17]['s'] = 22
        self.trans[17]['n'] = 12
        self.trans[18]['e'] = 19
        self.trans[18]['w'] = 17
        self.trans[18]['s'] = 23
        self.trans[18]['n'] = 13
        self.trans[19]['e'] = 20
        self.trans[19]['w'] = 18
        self.trans[19]['s'] = 24
        self.trans[19]['n'] = 14
        self.trans[20]['w'] = 19
        self.trans[20]['s'] = 25
        self.trans[20]['n'] = 15
        self.trans[21]['e'] = 22
        self.trans[21]['n'] = 16
        self.trans[22]['e'] = 23
        self.trans[22]['w'] = 21
        self.trans[22]['n'] = 17

        self.rewards = {} # 某状态下采取不同动作的收益
        for state in self.states:
            self.rewards[state] = {}
            for action in self.actions:
                self.rewards[state][action] = 0 # reward默认为0
                if state in self.trans and action in self.trans[state]: # 若trans中存在状态，且该状态下存在该动作
                    next_state = self.trans[state][action]
                    if next_state in self.terminate_states:
                        self.rewards[state][action] = self.terminate_states[next_state] # 满足上述条件，那么reward有特定的值

        self.pi = {} # 定义随机策略
        for state in self.trans:
            self.pi[state] = random.choice(list(self.trans[state].keys()))
        self.last_pi = self.pi.copy()

        self.value_space = {} # 定义价值表
        for state in self.states:
            self.value_space[state] = 0.0

    def get_random_action(self, state):
        self.pi[state] = random.choice(self.trans[state].keys())
        return self.pi[state]

    def transform(self, state, action):
        next_state = state
        state_reward = 0.0
        is_terminate = True
        return_info = {}

        if state in self.terminate_states: # 若为终止状态，直接返回
            return next_state, state_reward, is_terminate, return_info
        if state in self.trans: # 若状态位于状态转移矩阵，且该动作存在
            if action in self.trans[state]:
                next_state = self.trans[state][action]
        if state in self.rewards: # 若该状态位于rewards，且该动作存在
            if action in self.rewards[state]:
                state_reward = self.rewards[state][action]
        if not next_state in self.terminate_states: # 判断是否为终止状态
            is_terminate = False

        return next_state, state_reward, is_terminate, return_info

    def print_states(self):
        for state in self.states:
            if state in self.terminate_states:
                print('*', end='')
            else:
                print(round(self.value_space[state], 2), end='')
            if state % 5 == 0:
                print('|')


def monte_carlo_random(grid_mdp):
    data_list = []

    for _ in range(100000):
        one_sample_list = []
        state = random.choice(grid_mdp.states)

        # 如果直接是终态，放弃这一笔数据
        if state in grid_mdp.terminate_states:
            continue

        sample_end = False
        while sample_end != True:
            action = random.choice(list(grid_mdp.trans[state].keys()))
            next_state, state_reward, is_terminate, return_info = grid_mdp.transform(state, action)
            one_sample_list.append((state, action, state_reward))

            state = next_state
            sample_end = is_terminate
        data_list.append(one_sample_list)

    return data_list


def mc_value_func(data_list, grid_mdp):
    state_value_dic = {}

    for one_sample_list in data_list: # 遍历不同的episode, 计算每个state出现的次数，以及总体收益
        G = 0.0
        for idx in range(len(one_sample_list)-1, -1, -1): # 遍历一个episode中不同的状态，从后往前遍历，方便计算折扣后的收益
            one_sample = one_sample_list[idx]
            state, _, state_reward = one_sample[0], one_sample[1], one_sample[2]

            # 若是当前state第一次出现，加入到词典中
            if not state in state_value_dic:
                state_value_dic[state] = [0.0, 0.0] # 0维记录当前state出现的次数，1维记录当前state总共获得的收益

            G = state_reward + grid_mdp.gamma * G
            state_value_dic[state][0] += 1
            state_value_dic[state][1] += G

    for state in state_value_dic:
        if state in grid_mdp.value_space and state_value_dic[state][0] > 0: # 状态出现次数为0的不用记录
            grid_mdp.value_space[state] = state_value_dic[state][1] / state_value_dic[state][0]

    grid_mdp.print_states()


def mc_value_func_recursion(data_list, grid_mdp):
    state_value_dic = {}

    for one_sample_list in data_list:
        G = 0.0
        for idx in range(len(one_sample_list)-1, -1, -1):
            one_sample = one_sample_list[idx]
            state, _, state_reward = one_sample[0], one_sample[1], one_sample[2]

            if not state in state_value_dic:
                state_value_dic[state] = [0.0, 0.0]

            G = state_reward + grid_mdp.gamma * G
            state_value_dic[state][0] += 1
            state_value_dic[state][1] += (G - state_value_dic[state][1]) / state_value_dic[state][0]

    for state in state_value_dic:
        if state in state_value_dic:
            grid_mdp.value_space[state] = state_value_dic[state][1]
    grid_mdp.print_states()


grid_mdp = GridMdp()
data_list = monte_carlo_random(grid_mdp)
mc_value_func(data_list, grid_mdp)
mc_value_func_recursion(data_list, grid_mdp)

print(grid_mdp.value_space)
