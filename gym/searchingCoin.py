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
        pass
