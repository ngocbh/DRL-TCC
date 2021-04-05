import numpy as np
import time
import random

from environment import WRSNEnv

from utils import NetworkInput
from utils import gen_cgrg

def random_strategy(filepath):
    num_episode = 1
    # inp = NetworkInput.from_file(filepath)

    inp = gen_cgrg(20, 10, np.random.RandomState(1))

    env = WRSNEnv(inp)
    
    # seed = random.randint(1, 1000)
    # print(seed)
    # env.seed(383)
    for episode in range(num_episode):
        env.reset()

        for t in range(1000):
            env.render()
            action = env.action_space.sample()

            print(t, ":Go to ", action)

            next_state, reward, done, info = env.step(action)
            print(next_state)
            print(reward)
            print(done)
            
            time.sleep(2)
            if done:
                time.sleep(10)
                env.close()
                break

    print(env.net.network_lifetime)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    random_strategy('data/test/NIn1.json')

