import numpy as np
import time
import random

from environment import WRSNEnv

from utils import NetworkInput
from utils import gen_cgrg

def random_strategy(filepath):
    num_episode = 1
    # inp = NetworkInput.from_file(filepath)

    inp = gen_cgrg(20, 10, np.random.RandomState(3))

    env = WRSNEnv(inp)
    
    seed = random.randint(1, 1000)
    print(seed)
    env.seed(203)
    for episode in range(num_episode):
        env.reset()

        for t in range(1000):
            env.render()
            action = env.action_space.sample()

            next_state, reward, done, info = env.step(action)
            print("mc position: ", next_state[0][0], next_state[0][1])
            print("reward: ", reward)
            print("network_lifetime: ", env.net.network_lifetime)
            print("done: ", done)
            print("="*10 + '\n\n')
            
            time.sleep(1)
            if done:
                print(env.net.network_lifetime)
                input()
                env.close()
                break

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    random_strategy('data/test/NIn1.json')

