import os
import math
import numpy as np
import torch

from bisect import bisect_right
from torch.utils.data import Dataset

from utils.input import NetworkInput, Point
from utils.parameters import WrsnParameters as wp
from utils import dist



def gen_cgrg(num_sensors, num_targets, rand):
    """generate connected geometric ramdom graph.

    Parameters
    ----------
    num_sensors :
        num_sensors
    num_targets :
        num_targets
    rand :
        rand
    """

    def infer_new_point(A, B, gamma, d):
        alpha = math.atan2(B.y - A.y, B.x - A.x)
        alpha += gamma
        
        C_x = A.x + d * math.cos(alpha)
        C_y = A.y + d * math.sin(alpha)
        C = Point(C_x, C_y, 0)
        return C

    sink = Point(*wp.sink)
    depot = Point(*wp.depot)
    targets = []
    sensors = []

    epsilon = 10
    nodes = [sink]
    wheel = [epsilon]

    while len(sensors) < num_sensors:
        stake = bisect_right(wheel, rand.uniform(0, wheel[-1]))
        if stake != 0:
            angel = rand.uniform(0, 2 * math.pi)
            if angel > 2 * math.pi:
                angel = math.pi / 2 + (angel - 2 * math.pi) % math.pi
        else:
            angel = rand.uniform(0, 2 * math.pi)

        distance = rand.uniform(0.8 * wp.r_c, wp.r_c)
        
        new_sn = infer_new_point(nodes[stake], sink, angel, distance)
        nodes.append(new_sn)
        sensors.append(new_sn)
        
        d_to_sink = dist(new_sn, sink)
        wheel.append(wheel[-1] + d_to_sink + epsilon)

    while len(targets) < num_targets:
        stake = bisect_right(wheel, rand.uniform(0, wheel[-1]))
        if stake != 0:
            angel = rand.uniform(0, 2 * math.pi)
            if angel > 2 * math.pi:
                angel = math.pi / 2 + (angel - 2 * math.pi) % math.pi
        else:
            angel = rand.uniform(0, 2 * math.pi)
        distance = rand.uniform(0.9 * wp.r_s, wp.r_s)

        new_tg = infer_new_point(nodes[stake], sink, angel, distance)
        nodes.append(new_tg)
        targets.append(new_tg)

    inp = NetworkInput(wp.W, wp.H,
                       num_sensors=num_sensors,
                       num_targets=num_targets,
                       sink=sink,
                       depot=depot,
                       sensors=sensors,
                       targets=targets,
                       r_c=wp.r_c,
                       r_s=wp.r_s)
    assert inp.is_connected(), 'generated input is not connected'
    return inp
    
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

