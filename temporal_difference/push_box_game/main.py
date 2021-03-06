import gym
from temporal_difference.push_box_game.agent import QLearningAgent, SarsaAgent


if __name__ == "__main__":
    env = gym.make("pushBox-v0")

    RL = QLearningAgent(actions=env.action_space)
    # RL = SarsaAgent(actions=env.action_space)

    RL.train(env)
