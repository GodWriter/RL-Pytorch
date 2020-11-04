import gym
import numpy
import random
import logging

from gym import spaces
from gym.utils import seeding


class GridEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 2}

    def __init__(self):
        self.states = [1, 2, 3, 4, 5, 6, 7, 8]  # 状态空间
        self.actions = ['n', 'e', 's', 'w']  # 动作空间

        # 定义机器人可能的位置，提前计算好了
        self.x = [140, 220, 300, 380, 460, 140, 300, 460]
        self.y = [250, 250, 250, 250, 250, 150, 150, 150]

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

        self.seed()
        self.viewer = None
        self.state = None

    def step(self, action):
        state = self.state  # 系统当前状态

        if state in self.terminate_states:
            return state, 0, True, {}  # 下一时刻动作，回报，是否终止，调试信息

        key = "%d_%s" % (state, action)  # 状态和动作组合为状态转移字典的键值

        # 状态转移
        next_state = self.t[key] if key in self.t else state
        self.state = next_state

        is_terminal = True if next_state in self.terminate_states else False
        r = 0.0 if key not in self.rewards else self.rewards[key]

        return next_state, r, is_terminal, {}

    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        print("Current State:", self.state)

        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
            self.line1 = rendering.Line((100, 300), (500, 300))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (100, 100))
            self.line4 = rendering.Line((180, 300), (180, 100))
            self.line5 = rendering.Line((260, 300), (260, 100))
            self.line6 = rendering.Line((340, 300), (340, 100))
            self.line7 = rendering.Line((420, 300), (420, 100))
            self.line8 = rendering.Line((500, 300), (500, 100))
            self.line9 = rendering.Line((100, 100), (180, 100))
            self.line10 = rendering.Line((260, 100), (340, 100))
            self.line11 = rendering.Line((420, 100), (500, 100))

            # 创建第一个骷髅
            self.kulo1 = rendering.make_circle(40)
            self.kulo1.add_attr(rendering.Transform(translation=(140, 150)))
            self.kulo1.set_color(0, 0, 0)

            # 创建第二个骷髅
            self.kulo2 = rendering.make_circle(40)
            self.kulo2.add_attr(rendering.Transform(translation=(460, 150)))
            self.kulo2.set_color(0, 0, 0)

            # 创建金条
            self.gold = rendering.make_circle(40)
            self.gold.add_attr(rendering.Transform(translation=(300, 150)))
            self.gold.set_color(1, 0.9, 0)

            # 创建机器人
            self.robot = rendering.make_circle(30)
            self.robottrans = rendering.Transform()
            self.robot.add_attr(self.robottrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None

        # self.state从[1, 8]，而self.x/self.y坐标从[0, 7]，所以需要-1
        self.robottrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
