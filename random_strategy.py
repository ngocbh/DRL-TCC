import numpy as np
import time
import random
import torch

from torch.utils.data import DataLoader

from environment import WRSNEnv

from utils import WRSNDataset
from utils import DrlParameters as dp, WrsnParameters as wp
from utils import NetworkInput, Point
from utils import gen_cgrg, dist
from main import validate

def random_decision_maker(mc_state, depot_state, sn_state, mask):
    mask_ = mask.clone()
    n = len(sn_state)

    for i in range(0, n):
        d_mc_i = dist(Point(mc_state[0], mc_state[1]),
                      Point(sn_state[i, 0], sn_state[i, 1]))
        t_mc_i = d_mc_i / mc_state[6]
        d_i_bs = dist(Point(sn_state[i, 0], sn_state[i, 1]),
                      Point(depot_state[0], depot_state[1]))
        t_charge_i = (sn_state[i, 2] - sn_state[i, 4] + sn_state[i, 5] * t_mc_i) / \
                    (mc_state[5] - sn_state[i, 5])

        if mc_state[2] - mc_state[4] * d_mc_i - \
            (sn_state[i, 2] - sn_state[i, 4] + sn_state[i, 5] * (t_mc_i + t_charge_i)) \
            - mc_state[4] * d_i_bs < 0:
            mask_[i+1] = 0.0

    return np.random.choice(np.nonzero(mask_.cpu().numpy())[0]), 0.0

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)
    dataset = WRSNDataset(20, 10, 100, 1)
    data_loader = DataLoader(dataset, 1, False, num_workers=0)
    # wp.from_file('./configs/mc_20_10_4_small.yml')
    wp.from_file('./configs/mc_20_10_2_small.yml')
    ret = validate(data_loader, random_decision_maker, wp=wp, render=False, verbose=False, normalize=False)
    print(ret)

