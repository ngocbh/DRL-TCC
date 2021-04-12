import os
import math
import numpy as np
import torch

from bisect import bisect_right
from torch.utils.data import Dataset

from utils.input import NetworkInput, Point
from utils.parameters import WrsnParameters as wp
from utils import dist


def gen_cgrg(num_sensors, num_targets, rand=np.random.RandomState()):
    num_trial = 0
    while True:
        num_trial += 1
        data = rand.uniform(size=(num_sensors + num_targets, 2))
        sink = Point(**wp.sink)
        depot = Point(**wp.depot)
        sensors = [Point(x * wp.W, y * wp.H) for x, y in data[:num_sensors]]
        targets = [Point(x * wp.W, y * wp.H) for x, y in data[num_sensors:]]

        inp = NetworkInput(wp.W, wp.H,
                           num_sensors=num_sensors,
                           num_targets=num_targets,
                           sink=sink,
                           depot=depot,
                           sensors=sensors,
                           targets=targets,
                           r_c=wp.r_c,
                           r_s=wp.r_s)
        if inp.is_connected():
            return data[:num_sensors], data[num_sensors:], num_trial
    
class WRSNDataset(Dataset):
    def __init__(self, num_sensors, num_targets, num_samples=1e4, seed=None):
        super(WRSNDataset, self).__init__()

        self.rand = np.random.RandomState(seed)

        self.num_sensors = num_sensors
        self.num_targets = num_targets
        self.num_samples = num_samples

        self.dataset = []
        self.num_trial = 0

        for _ in range(int(num_samples)):
            sensors, targets, nt = gen_cgrg(num_sensors, num_targets, self.rand)
            self.dataset.append((sensors, targets))
            self.num_trial += nt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.dataset[index]


def gen_cgrg_layer_based(num_sensors, num_targets, rand=np.random.RandomState()):
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

    level = 0
    num_sn_layer = 0
    parent = []
    current_layer = [sink]

    while len(sensors) < num_sensors:
        if num_sn_layer == 0:
            level += 1
            parent = [node for node in current_layer]
            current_layer = []
            a = 2 * level
            b = 2 ^ level +  3
            high = min(num_sensors - len(sensors), b)
            low = min(high, a)
            num_sn_layer = rand.randint(low, high+1)

        stake = rand.randint(0, len(parent))
        if stake != 0:
            # angel = rand.uniform(0, 2 * math.pi / 2)
            angel = rand.uniform(math.pi/2, 3/2 * math.pi)
            # if angel > 2 * math.pi:
                # angel = math.pi / 2 + (angel - 2 * math.pi) % math.pi
        else:
            angel = rand.uniform(0, 2 * math.pi)

        distance = rand.uniform(0.7 * wp.r_c, wp.r_c)
        
        new_sn = infer_new_point(parent[stake], sink, angel, distance)
        if 0 < new_sn.x < wp.W and 0 < new_sn.y < wp.H:
            nodes.append(new_sn)
            sensors.append(new_sn)
            current_layer.append(new_sn)
            num_sn_layer -= 1
        

    while len(targets) < num_targets:
        stake = rand.randint(0, len(nodes))
        if stake != 0:
            angel = rand.uniform(0, 2 * math.pi)
            if angel > 2 * math.pi:
                angel = math.pi / 2 + (angel - 2 * math.pi) % math.pi
        else:
            angel = rand.uniform(0, 2 * math.pi)

        distance = rand.uniform(0.5 * wp.r_s, wp.r_s)

        new_tg = infer_new_point(nodes[stake], sink, angel, distance)
        if 0 < new_tg.x < wp.W and 0 < new_sn.y < wp.H:
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
