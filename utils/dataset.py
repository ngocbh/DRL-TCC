import os
import numpy as np
import torch

from torch.utils.data import Dataset

from utils.input import NetworkInput, Point
from utils.parameters import WrsnParameters as wp

def gen_cgrg(num_sensors, num_targets, rand):
    return 0
    
class WRSNDataset(Dataset):
    def __init__(self, num_sensors, num_targets, num_samples=1e4, seed=None):
        super(WRSNDataset, self).__init__()

        self.rand = np.random.RandomState(seed)

        self.num_sensors = num_sensors
        self.num_targets = num_targets
        self.num_samples = num_samples

        self.dataset = []
        for _ in range(num_samples):
            inp = gen_cgrg(num_samples, num_targets, self.rand)
            self.dataset.append(inp)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.dataset[index]

