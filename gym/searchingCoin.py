import gym
import numpy
import random
import logging

from gym import spaces


logger = logging.getLogger(__name__)


class GridEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 2}

    def __init__(self):
        self.states =[1, 2, 3, 4, 5, 7, 8] # 状态空间
        self.actions = ['n', 'e', 's', 'w'] # 动作空间

        # 定义终止状态
        self.terminate_states = dict()
        self.terminate_states[6] = 1
        self.terminate_states[7] = 1
        self.terminate_states[8] = 1

        # 定义reward
        self.rewards = dict()
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        # 定义状态转移
        self.t = dict()
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4

    def _step(self, action):
        state = self.state # 系统当前状态

        if state in self.terminate_states:
            return state, 0, True, {} # 下一时刻动作，回报，是否终止，调试信息

        key = "%d_%s" % (state, action) # 状态和动作组合为状态转移字典的键值

        # 状态转移
        next_state = self.t[key] if key in self.t else state
        self.state = next_state

        is_terminal = True if next_state in self.terminate_states else False
        r = 0.0 if key not in self.rewards else self.rewards[key]

        return next_state, r, is_terminal, {}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(width=600, height=400)

            # 创建网格世界
            pass
