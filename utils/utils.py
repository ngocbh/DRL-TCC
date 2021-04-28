from collections import namedtuple

import math
import os
import torch
import pickle

from utils.logger import logger

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

Point = namedtuple('Point', ['x', 'y', 'z'], defaults=[0, 0, 0])


def dist(p1: Point, p2: Point):
    return math.dist(list(p1), list(p2))


def normalize(x, low, high):
    return (x - low) / high


def bound(x, low, high):
    return min(max(x, low), high)


def pdump(x, name, outdir='.'):
    with open(os.path.join(outdir, name), mode='wb') as f:
        pickle.dump(x, f)


def pload(name, outdir='.'):
    with open(os.path.join(outdir, name), mode='rb') as f:
        return pickle.load(f)

