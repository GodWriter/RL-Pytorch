import gym
import numpy as np

from gym.utils import seeding
from gym.envs.classic_control import rendering


class PushBoxEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 1}

    def __init__(self):
        self.reward = np.array([[-100, -100, -100, -100, -100],
                                [-100,    0,    0,    0, -100],
                                [-100,    0, -100,    0, -100],
                                [-100,    0, -100,    0, -100],
                                [-100,    0,  100,    0, -100]]).T

        self.walls = [(0, 1), (0, 3), (2, 2), (2, 3)] # 墙壁坐标
        self.dest_position = (2, 4) # 终点站坐标
        self.work_position = (2, 0) # 推箱人的坐标
        self.box_position = (2, 1) # 箱子的坐标
        self.action_space = ((1, 0), (-1, 0), (0, 1), (0, -1)) # 动作空间：上下左右
        self.viewer = None
        self.__state = None
        self.seed()

    def _is_terminal(self, state):
        box_position = state[2:]
        if self.reward[box_position] != 0: # -100和100的奖励都是终态，要么靠墙输了，要么推到终点了
            return True
        return False

    def _move_ok(self, position):
        if tuple(position) not in self.walls and (position >= 0).all() and (position < 5).all(): # 因为position代表下一个位置，在边界的箱子会超出格子，不满足条件
            return True
        return False

    def _trans_make(self, state, action):
        work_position = np.array(state[:2]) # 推箱人坐标
        box_position = np.array(state[2:]) # 箱子坐标

        action = np.array(action)
        next_work_position = work_position + action

        if self._move_ok(next_work_position): # 判断推箱人是否还可以移动，即坐标是否合理；
            work_position = next_work_position # 若坐标合理，改变当前人的坐标为下一步坐标

            if (next_work_position == box_position).all(): # 只有人往箱子方向走了，箱子才会被推动，人的坐标应该和箱子坐标一致；否则箱子不动
                next_box_position = box_position + action # 只有人推箱子了，箱子的坐标才改变
                if self._move_ok(next_box_position): # 判断箱子的坐标是否合理，合理的话更改当前箱子坐标
                    box_position = next_box_position
                else:
                    work_position -= action # 虽然箱子在前方但是只是挡住了人的路线, 需要将已经移动的人还原

        return tuple(np.hstack((work_position, box_position)))

    def _reward(self, state):
        box_position = state[2:]
        return self.reward[box_position]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()

    def transform(self, state, action):
        if self._is_terminal(state):
            return state, self._reward(state), True, {}

        next_state = self._trans_make(state, action) # 状态转移
        r = self._reward(next_state) # 计算回报

        is_terminal = False # 判断下一状态是否为终态
        if self._is_terminal(next_state):
            is_terminal = True

        return next_state, r, is_terminal, {}

    def step(self, action):
        state = self.__state # 系统当前状态
        next_state, r, is_terminal, _ = self.transform(state, action) # 调用transform状态转移
        self.__state = next_state # 状态转移

        return next_state, r, is_terminal, {}

    def reset(self):
        self.__state = tuple(np.hstack((self.work_position, self.box_position)))
        return self.__state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        unit = 50 # 格子的单位长度
        screen_width = 5 * unit
        screen_height = 5 * unit

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 创建迷宫的网格
            for c in range(5):
                line = rendering.Line((0, c*unit), (screen_width, c*unit))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)
            for r in range(5):
                line = rendering.Line((r*unit, 0), (r*unit, screen_height))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)

            # 创建迷宫中的墙壁
            for x, y in self.walls:
                # v提供了一个闭环矩形
                r = rendering.make_polygon(v=[[x * unit, y * unit],
                                              [(x + 1) * unit, y * unit],
                                              [(x + 1) * unit, (y + 1) * unit],
                                              [x * unit, (y + 1) * unit],
                                              [x * unit, y * unit]])
                r.set_color(0, 0, 0)
                self.viewer.add_geom(r)

            # 创建终点
            d_x, d_y = self.dest_position
            dest = rendering.make_polygon(v=[[d_x * unit, d_y * unit],
                                             [(d_x + 1) * unit, d_y * unit],
                                             [(d_x + 1) * unit, (d_y + 1) * unit],
                                             [d_x * unit, (d_y + 1) * unit],
                                             [d_x * unit, d_y * unit]])
            dest_trans = rendering.Transform()
            dest.add_attr(dest_trans)
            dest.set_color(1, 0, 0)
            self.viewer.add_geom(dest)

            # 创建推箱子的人
            self.work = rendering.make_circle(20)
            self.work_trans = rendering.Transform()
            self.work.add_attr(self.work_trans)
            self.work.set_color(0, 1, 0)
            self.viewer.add_geom(self.work)

            # 创建箱子
            self.box = rendering.make_circle(20)
            self.box_trans = rendering.Transform()
            self.box.add_attr(self.box_trans)
            self.box.set_color(0, 0, 1)
            self.viewer.add_geom(self.box)

        if self.__state is None:
            return None

        # 设置推箱人和箱子的初始位置
        w_x, w_y = self.__state[:2]
        b_x, b_y = self.__state[2:]
        self.work_trans.set_translation(w_x*unit + unit / 2, w_y*unit + unit /2)
        self.box_trans.set_translation(b_x*unit + unit/2, b_y*unit + unit/2)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


if __name__ == '__main__':
    env = PushBoxEnv()
    env.reset()
    env.render()
    env.step((-1, 0))
    env.render()
