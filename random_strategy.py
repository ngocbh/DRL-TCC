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

def validate(data_loader, save_dir='.', render=False, verbose=False, max_step=None):
    times = [0]
    net_lifetimes = []
    mc_travel_dists = []
    mean_aggregated_ecrs = []
    mean_node_failures = []

    for data in data_loader:
        sensors, targets = data

        env = WRSNEnv(sensors=sensors.squeeze(), 
                      targets=targets.squeeze(), 
                      normalize=False)
        env.reset()

        mask = np.ones(env.action_space.n)
        ecrs = []
        node_failures = []

        max_step = max_step or dp.max_step
        for _ in range(max_step):
            if render:
                env.render()
                
            action = np.random.choice(np.nonzero(mask)[0])

            mask[env.last_action] = 1
            (mc_state, depot_state, sn_state), reward, done, _ = env.step(action)
            mask[env.last_action] = 0

            if verbose:
                print("mc_state\n", mc_state)
                print("depot_state\n", depot_state)
                print("sn_state\n", sn_state)

            ecrs.append(env.net.aggregated_ecr)
            node_failures.append(env.net.node_failures)

            if done:
                if render:
                    env.render()
                    input()

                env.close()
                break
                
            if render:
                time.sleep(0.7)

        net_lifetimes.append(env.get_network_lifetime())
        mc_travel_dists.append(env.get_travel_distance())
        mean_aggregated_ecrs.append(np.mean(ecrs))
        mean_node_failures.append(np.mean(node_failures))
    
    ret = {}
    ret['lifetime_mean'] = np.mean(net_lifetimes)
    ret['lifetime_std'] = np.std(net_lifetimes)
    ret['travel_dist_mean'] = np.mean(mc_travel_dists)
    ret['travel_dist_std'] = np.std(mc_travel_dists)
    ret['aggregated_ecr_mean'] = np.mean(mean_aggregated_ecrs)
    ret['aggregated_ecr_std'] = np.std(mean_aggregated_ecrs)
    ret['node_failures_mean'] = np.mean(mean_node_failures)
    ret['node_failures_std'] = np.std(mean_node_failures)

    return ret

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    dataset = WRSNDataset(20, 10, 1000, 1)
    data_loader = DataLoader(dataset, 1, False, num_workers=0)
    validate(data_loader, render=True, verbose=True)

