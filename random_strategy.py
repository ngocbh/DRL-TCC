import numpy as np
import time
import random
import torch

from torch.utils.data import DataLoader

from environment import WRSNEnv

from utils import WRSNDataset
from utils import DrlParameters as dp
from utils import NetworkInput
from utils import gen_cgrg
from main import validate

def random_decision_maker(mc_state, depot_state, sn_state, mask):
    print(torch.nonzero(mask).squeeze())
    return np.random.choice(torch.nonzero(mask).squeeze()), 1.0

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    dataset = WRSNDataset(20, 10, 1000, 1)
    data_loader = DataLoader(dataset, 1, False, num_workers=0)
    validate(data_loader, random_decision_maker, render=True, verbose=True)

