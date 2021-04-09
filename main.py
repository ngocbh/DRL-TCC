import random
import os
import torch
import time
import numpy as np
import click

from torch.utils.data import DataLoader

from model import MCActor, Critic
from environment import WRSNEnv
from utils import NetworkInput, WRSNDataset
from utils import Config, DrlParameters as dp, WrsnParameters as wp
from utils import logger, gen_cgrg, device



@click.command()
@click.option(
    '--num_sensors', '-ns', default=20, type=int
)
@click.option(
    '--num_targets', '-nt', default=10, type=int
)
@click.option(
    '--config', '-cf', default=None, type=str
)
@click.option(
    '--checkpoint', '-cp', default=None, type=str
)
@click.option(
    '--save_dir', '-sd', default='checkpoints', type=str
)
@click.option(
    '--seed', default=None, type=int
)
def train(num_sensors=20, num_targets=10, config=None,
          checkpoint=None, save_dir='checkpoints', seed=None):
    # logger.info("Training problem with %d sensors %d targets (checkpoint: %s) ()")
    if config is not None:
        wp.from_file(config)
        dp.from_file(config)

    train_data = WRSNDataset(num_sensors, num_targets, dp.train_size, seed)
    valid_data = WRSNDataset(num_sensors, num_targets, dp.valid_size, seed + 1)

    actor = MCActor(dp.MC_INPUT_SIZE, 
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    critic = Critic(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size).to(device)

    if checkpoint is not None:
        path = os.path.join(checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    train_loader = DataLoader(train_data, dp.batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, dp.batch_size, False, num_workers=0)

    for epoch in range(dp.num_epoch):
        actor.train()
        critic.train()

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):
            pass

def test():
    # filepath = 'data/test/NIn1.json'
    # inp = NetworkInput.from_file(filepath)
    dataset = WRSNDataset(20, 10)
    print(dataset.num_trial)

    inp = dataset[0]
    env = WRSNEnv(inp, normalize=True)

    seed = random.randint(1, 1000)
    logger.info("Seeding env: %d" % seed)
    env.seed(seed)

    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    critic = Critic(dp.MC_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size).to(device)

    env.reset()
    env.render()

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
    train()
    # test()
    # gen_cgrg(20, 10, np.random.RandomState(1))

