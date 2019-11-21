#https://www.statworx.com/at/blog/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow/
import numpy as np
from srcScrCapMeth.Screencap import screencap
from srcScrCapMeth.Interaction import take_action


class Environment:
    """ A simple environment skeleton """
    def __init__(self):
        # Initializes the environment
        self.action_space = ['X', 'Z', 'RIGHT', 'LEFT', 'DOWN']

    def step(self, action):
        # Changes the environment based on agents action
        take_action(action)


        next_state = 0
        reward = 0
        done = False
        info = 0
        return next_state, reward, done, info

    def reset(self):
        # Resets the environment to its initial state
        pass

    def render(self):
        # Show the state of the environment on screen
        pass

    def close(self):
        # Close the environment
        pass

