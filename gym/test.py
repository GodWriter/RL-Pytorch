import gym
import time

env = gym.make("searchCoin-v0")

for i in range(5):
    env.reset()
    env.render()
    time.sleep(1)

env.close()
