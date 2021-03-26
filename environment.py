import gym


from utils import WrsnParameters as wp
from utils import NetworkInput, Point

class WRSNEnv(gym.Env):
    """WRSNEnv.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, inp: NetworkInput):
        self.inp = inp

    def step(self, action):
        pass

    def t_step(self, action, t):
        pass

    def render(self, mode):
        raise NotImplementedError

if __name__ == '__main__':
    pass
