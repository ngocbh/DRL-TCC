import numpy as np
import random

from environment import WRSNEnv
from utils import NetworkInput

def random_strategy(filepath):
    num_episode = 1
    inp = NetworkInput.from_file(filepath)
    env = WRSNEnv(inp)
    
    # seed = random.randint(1, 1000)
    # print(seed)
    env.seed(383)
    for episode in range(num_episode):
        env.reset()

        for t in range(1000):
            action = env.action_space.sample()

            print(t, ":Go to ", action)

            next_state, reward, done, info = env.step(action)
            print(next_state)
            print(reward)
            print(done)
            
            if done:
                break

    print(env.net.network_lifetime)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    random_strategy('data/test/NIn1.json')

