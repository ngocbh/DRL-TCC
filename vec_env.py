import numpy as np

from torch.utils.data import DataLoader

from stable_baselines3.common.vec_env import (DummyVecEnv,
                                              SubprocVecEnv,
                                              VecEnvWrapper)

from environment import WRSNEnv
from utils import WRSNDataset
from utils import WrsnParameters as wp, DrlParameters as dp

def make_env(sensors,
             targets,
             rank,
             seed=None,
             normalize=True):
    def _make():
        env = WRSNEnv(sensors=sensors.squeeze(), 
                      targets=targets.squeeze(), 
                      normalize=normalize)

        env.seed(seed + rank)
        return env

    return _make


def make_vec_envs(sensors_batch,
                  targets_batch,
                  log_dir,
                  allow_early_resets,
                  seed=None):
    envs = []
    for i, (sensors, targets) in enumerate(zip(sensors_batch, targets_batch)):
        env = make_env(sensors, targets, i, seed, normalize=True)
        envs.append(env)

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    
    return envs

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    dataset = WRSNDataset(20, 10, 32, 1)
    data_loader = DataLoader(dataset, 16, False, num_workers=0)
    sensors_batch, targets_batch = next(iter(data_loader))
    envs = make_vec_envs(sensors_batch, targets_batch, None, True, 1)
    mc_state, depot_state, sn_state = envs.reset()
    
    for step in range(1):
        actions = [envs.action_space.sample() for _ in range(16)]
        envs.step_async(actions)
        (mc_state, depot_state, sn_state), rewards, dones, infos = envs.step_wait()
        print(rewards.shape)
        print(dones)
        print(mc_state.shape, depot_state.shape, sn_state.shape)