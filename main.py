import random
import torch
import numpy as np

from model import MCActor, Critic
from environment import WRSNEnv
from utils import NetworkInput
from utils import DrlParameters as dp
from utils import logger, gen_cgrg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    pass


def test():
    # filepath = 'data/test/NIn1.json'
    # inp = NetworkInput.from_file(filepath)

    inp = gen_cgrg(20, 10, np.random.RandomState(1))
    env = WRSNEnv(inp, normalize=True)

    seed = random.randint(1, 1000)
    logger.info("Seeding env: %d" % seed)
    env.seed(seed)

    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.HIDDEN_SIZE,
                    dropout=dp.DROPOUT,
                    device=device).to(device)

    critic = Critic(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.HIDDEN_SIZE).to(device)

    env.reset()
    env.render()
    input()
    return

    mc_state, sensors_state = env.get_normalized_state()

    mc_state = torch.Tensor(mc_state, device=device)
    sensors_state = torch.Tensor(sensors_state, device=device)
    mc_state = mc_state.unsqueeze(0)
    sensors_state = sensors_state.unsqueeze(0)

    batch_size, sequence_size, input_size = sensors_state.size()

    actor(mc_state, sensors_state)
    critic(mc_state, sensors_state)


if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)
    test()
