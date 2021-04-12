from collections import namedtuple
from numba import jit

import math
import torch
import logging

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

Point = namedtuple('Point', ['x', 'y', 'z'], defaults=[0, 0, 0])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

def dist(p1: Point, p2: Point):
    return math.dist(list(p1), list(p2))

def normalize(x, low, high):
    return (x - low) / high

def bound(x, low, high):
    return min(max(x, low), high)
