import random


class GriDMdp:
    def __init__(s):
        s.gamma = 0.9
        s.states = range(1, 26)
        s.actions = ['n', 'e', 's', 'w']
        s.terminate_states = {15: 1.0, 4: -1.0, 9: -1.0, 11: -1.0, 12: -1.0, 23: -1.0, 24: -1.0, 25: -1.0}
        s.trans = {}
        for state in s.states:
            if not state in s.terminate_states:
                s.trans[state] = {}

        s.trans[1]['e'] = 2
        s.trans[1]['s'] = 6
        s.trans[2]['e'] = 3
        s.trans[2]['w'] = 1
        s.trans[2]['s'] = 7
        s.trans[2]['s'] = 7
        s.trans[3]['e'] = 4
        s.trans[3]['w'] = 2
        s.trans[3]['s'] = 8
        s.trans[5]['w'] = 4
        s.trans[5]['s'] = 10
        s.trans[6]['e'] = 7
        s.trans[6]['s'] = 11
        s.trans[6]['n'] = 1
        s.trans[7]['e'] = 8
        s.trans[7]['w'] = 6
        s.trans[7]['s'] = 12
        s.trans[7]['n'] = 2
        s.trans[8]['e'] = 9
        s.trans[8]['w'] = 7
        s.trans[8]['s'] = 13
        s.trans[8]['n'] = 3
        s.trans[10]['w'] = 9
        s.trans[10]['s'] = 15
        s.trans[13]['e'] = 14
        s.trans[13]['w'] = 12
        s.trans[13]['s'] = 18
        s.trans[13]['n'] = 8
        s.trans[14]['e'] = 15
        s.trans[14]['w'] = 13
        s.trans[14]['s'] = 19
        s.trans[14]['n'] = 9
        s.trans[16]['e'] = 17
        s.trans[16]['s'] = 21
        s.trans[16]['n'] = 11
        s.trans[17]['e'] = 18
        s.trans[17]['w'] = 16
        s.trans[17]['s'] = 22
        s.trans[17]['n'] = 12
        s.trans[18]['e'] = 19
        s.trans[18]['w'] = 17
        s.trans[18]['s'] = 23
        s.trans[18]['n'] = 13
        s.trans[19]['e'] = 20
        s.trans[19]['w'] = 18
        s.trans[19]['s'] = 24
        s.trans[19]['n'] = 14
        s.trans[20]['w'] = 19
        s.trans[20]['s'] = 25
        s.trans[20]['n'] = 15
        s.trans[21]['e'] = 22
        s.trans[21]['n'] = 16
        s.trans[22]['e'] = 23
        s.trans[22]['w'] = 21
        s.trans[22]['n'] = 17

        s.rewards = {}
        for state in s.states:
            s.rewards[state] = {}
            for action in s.actions:
                s.rewards[state][action] = 0
                if state in s.trans and action in s.trans[state]:
                    next_state = s.trans[state][action]
                    if next_state in s.terminate_states:
                        s.rewards[state][action] = s.terminate_states[next_state]
        s.pi = {}
        for state in s.trans:
            s.pi[state] = random.choice(list(s.trans[state].keys()))
        s.last_pi = s.pi.copy()

        s.v = {}
        for state in s.states:
            s.v[state] = 0.0

    def get_random_action(s, state):
        s.pi[state] = random.choice(s.trans[state].keys())
        return s.pi[state]

    def transform(s, state, action):
        next_state = state
        state_reward = 0
        is_terminate = True
        return_info = {}

        if state in s.terminate_states:
            return next_state, state_reward, is_terminate, return_info
        if state in s.trans:
            if action in s.trans[state]:
                next_state = s.trans[state][action]
        if state in s.rewards:
            if action in s.rewards[state]:
                state_reward = s.rewards[state][action]
        if not next_state in s.terminate_states:
            is_terminate = False
        return next_state, state_reward, is_terminate, return_info

    def print_states(s):
        for state in s.states:
            if state in s.terminate_states:
                print("*")
            else:
                print(round(s.v[state], 2))
            if state % 5 == 0:
                print("|")


def monte_carlo_random(grid_mdp):
    data_list = []
    for iter_idx in range(100000):
        one_sample_list = []
        state = random.choice(grid_mdp.states)
        if state in grid_mdp.terminate_states:
            continue
        sample_end = False
        while sample_end != True:
            # choose random strategy
            action = random.choice(list(grid_mdp.trans[state].keys()))
            next_state, state_reward, is_terminate, return_info = grid_mdp.transform(state, action)
            one_sample_list.append((state, action, state_reward))
            state = next_state
            sample_end = is_terminate
        data_list.append(one_sample_list)
    return data_list


def mc_value_func(data_list, grid_mdp):
    state_value_dic = {}
    for one_sample_list in data_list:
        G = 0.0
        print("-----------------------")
        print(one_sample_list)
        for idx in range(len(one_sample_list) - 1, -1, -1):
            one_sample = one_sample_list[idx]
            state = one_sample[0]
            action = one_sample[1]
            state_reward = one_sample[2]
            if not state in state_value_dic:
                state_value_dic[state] = [0.0, 0.0]
            G = state_reward + grid_mdp.gamma * G
            state_value_dic[state][0] += 1
            state_value_dic[state][1] += G
            print(idx, one_sample, G)
            print(state_value_dic)
    for state in state_value_dic:
        if state in grid_mdp.v and state_value_dic[state][0] > 0:
            grid_mdp.v[state] = state_value_dic[state][1] / state_value_dic[state][0]
    grid_mdp.print_states()


def mc_value_func_recursion(data_list, grid_mdp):
    state_value_dic = {}
    for one_sample_list in data_list:
        G = 0.0
        print("-----------------------")
        print(one_sample_list)
        for idx in range(len(one_sample_list) - 1, -1, -1):
            one_sample = one_sample_list[idx]
            state = one_sample[0]
            action = one_sample[1]
            state_reward = one_sample[2]
            if not state in state_value_dic:
                state_value_dic[state] = [0.0, 0.0]
            G = state_reward + grid_mdp.gamma * G
            state_value_dic[state][0] += 1
            state_value_dic[state][1] += (G - state_value_dic[state][1]) / state_value_dic[state][0]
            print(idx, one_sample, G)
            print(state_value_dic)
    for state in state_value_dic:
        if state in grid_mdp.v:
            grid_mdp.v[state] = state_value_dic[state][1]
    grid_mdp.print_states()


grid_mdp = GriDMdp()
data_list = monte_carlo_random(grid_mdp)
mc_value_func(data_list, grid_mdp)
mc_value_func_recursion(data_list, grid_mdp)

