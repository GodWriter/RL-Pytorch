import gym
import time

env = gym.make("pushBox-v0")
env.reset()
env.render()

time.sleep(10)