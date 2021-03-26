import math

from utils.input import Point

def dist(p1: Point, p2: Point):
    return math.dist(list(p1), list(p2))
