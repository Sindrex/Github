#https://www.statworx.com/at/blog/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow/

class Agent:
    """ A simple agent """
    def __init__(self):
        pass

    def action(self, state):
        if state > 0:
            return 1
        else:
            return 0

