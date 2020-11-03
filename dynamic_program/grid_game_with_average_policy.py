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

        # 判断更新后的状态是否在状态空间中，true更新状态，false保持原始状态
        if (state_ >= 0).all() and (state_ < 4).all():
            state_ = tuple(state_)
        else:
            state_ = state

        return state_

    def learn(self):
        """
        Average Policy.
        :return:
        """

        while True:
            print(self.value_space)
            v_s_ = self.value_space.copy()

            for state in self.state_space:
                q_s_a = pd.Series() # 记录当前状态下，采取不同动作能获得的价值

                for action in self.action_space:
                    state_ = self.transform(state, action)
                    if state_ in self.terminal_states:
                        q_s_a[action] = 0
                    elif state_ != state:
                        q_s_a[action] = v_s_[state_]
                    else:
                        q_s_a[action] = v_s_[state]

                # 更新价值表
                self.value_space[state] = sum([self.policy[action] * (self.reward + self.gamma * q_s_a[action])
                                               for action in self.action_space])

            # self.value_space代表当前价值表，v_s_代表上一价值表
            if (np.abs(v_s_ - self.value_space) < 1e-8).all():
                break

        return self.value_space


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

    # 开始
    mdp.learn()


if __name__ == '__main__':
    main()
