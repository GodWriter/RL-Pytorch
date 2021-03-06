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
        self.policy = pd.Series(data=np.random.choice(self.action_space, size=(len(self.state_space))), index=self.state_space)

    def transform(self, state, action):
        dir = self.__action_dir[action]
        state_ = np.array(state) + dir

        # 判断更新后的状态是否在状态空间中，true更新状态，false保持原始状态
        if (state_ >= 0).all() and (state_ < 4).all():
            state_ = tuple(state_)
        else:
            state_ = state

        return state_

    def policy_evaluate(self):
        while True:
            v_s_ = self.value_space.copy()
            # print(self.value_space)

            for state in self.state_space:
                action = self.policy[state]
                state_ = self.transform(state, action)

                if state_ in self.terminal_states:
                    self.value_space[state] = self.reward + 0.0
                elif state_ != state:
                    self.value_space[state] = self.reward + self.gamma * v_s_[state_]
                else:
                    self.value_space[state] = self.reward + self.gamma * v_s_[state]

            if (np.abs(v_s_ - self.value_space) < 1e-8).all():
                break

        return self.value_space

    def policy_improve(self):
        # pd.Series()初始化一定要注意，给定大小否则后面索引会报错
        policy_ = pd.Series(data=np.random.choice(self.action_space, size=(len(self.state_space))), index=self.state_space)

        for state in self.state_space:
            q_s_a = pd.Series()
            for action in self.action_space:
                state_ = self.transform(state, action)
                if state_ in self.terminal_states:
                    q_s_a[action] = self.reward
                else:
                    q_s_a[action] = self.reward + self.gamma * self.value_space[state_]

            # 可能出现多个最大值，为避免重复选择同一个，随机选择
            max_v = q_s_a[q_s_a == q_s_a.max()].index
            policy_[state] = np.random.choice(max_v)

        return policy_

    def policy_iterate(self):
        while True:
            print(self.policy)
            v_s = self.policy_evaluate()
            policy_ = self.policy_improve()

            if (policy_ == self.policy).all():
                break
            else:
                self.policy = policy_

        return self.policy


def main():
    # 定义状态空间
    state_space = [(i, j) for i in range(4) for j in range(4)]
    state_space.remove((0, 0))
    state_space.remove((3, 3))

    # 定义动作空间
    action_space = ["n", "s", "w", "e"]

    # 定义mdp，注意gamma<1，否则self.value_space无法收敛
    mdp = GridMDP(state_space=state_space,
                  action_space=action_space,
                  reward=-1,
                  gamma=0.8)

    # 开始
    mdp.policy_iterate()


if __name__ == '__main__':
    main()
