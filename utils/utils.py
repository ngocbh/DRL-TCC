import math
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        # logging.StreamHandler()
    ]
)

logger = logging.getLogger()

from utils.input import Point

def dist(p1: Point, p2: Point):
    return math.dist(list(p1), list(p2))

def normalize(x, low, high):
    return (x - low) / high
