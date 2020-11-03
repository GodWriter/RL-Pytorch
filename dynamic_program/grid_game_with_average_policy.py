import pandas as pd
import numpy as np


class GridMDP:
    def __init__(self,
                 state_space,
                 action_space,
                 reward,
                 gamma):

        self.state_space = state_space
        self.action_space = action_space
        self.reward = reward
        self.gamma = gamma

        self.__action_dir = pd.Series(data=[np.array((-1, 0)),
                                            np.array((1, 0)),
                                            np.array((0, -1)),
                                            np.array((0, 1))],
                                      index=self.action_space)
        self.terminal_states = [(0, 0), (3, 3)]

        # 定义值函数表, 策略表
        self.value_space = pd.Series(np.zeros((len(state_space))), index=self.state_space)
        self.policy = pd.Series(data=0.25 * np.ones(shape=(4, )), index=self.action_space)

    def transform(self, state, action):
        dir = self.__action_dir[action]
        state_ = np.array(state) + dir

        if (state_ >= 0).all() and (state_ < 4).all():
            state_ = tuple(state_)
        else:
            state_ = state

        return state_

    def learn(self):
        pass


def main():
    # 定义状态空间
    state_space = [(i, j) for i in range(4) for j in range(4)]
    state_space.remove((0, 0))
    state_space.remove((3, 3))

    # 定义动作空间
    action_space = ["n", "s", "w", "e"]

    # 定义mdp
    mdp = GridMDP(state_space=state_space,
                  action_space=action_space,
                  reward=-1,
                  gamma=1)


if __name__ == '__main__':
    main()
