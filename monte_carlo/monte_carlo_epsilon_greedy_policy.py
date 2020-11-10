import random
import numpy as np


class GridMdp:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.states = range(1, 26)
        self.actions = ['n', 'e', 's', 'w']
        self.terminate_states = {15: 1.0, 4: -1.0, 9: -1.0, 11: -1.0, 12: -1.0, 23: -1.0, 24: -1.0, 25: -1.0}

        self.trans = {}
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

        self.rewards = {}
        for state in self.states:
            self.rewards[state] = {}
            for action in self.actions:
                self.rewards[state][action] = 0.0
                if state in self.trans and action in self.trans[state]:
                    next_state = self.trans[state][action]
                    if next_state in self.terminate_states:
                        self.rewards[state][action] = self.terminate_states[next_state]

        self.pi = {}
        for state in self.trans:
            self.pi[state] = random.choice(list(self.trans[state].keys()))
        self.last_pi = self.pi.copy()

        self.value_space = {}
        for state in self.states:
            self.value_space[state] = 0.0

