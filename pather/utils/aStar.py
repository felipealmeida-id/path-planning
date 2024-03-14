import math
import numpy as np
from astar import AStar


class PathTraverser(AStar):

    area = []
    width = 0
    height = 0
    deltas = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    def __init__(self,area):
        self.area = area
        self.height = len(area)
        self.width = len(area[0])

    def heuristic_cost_estimate(self, n1, n2):
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        return 1

    def neighbors(self, node):
        x, y = node
        result = []
        for (dx,dy) in self.deltas:
            new_x = x+dx
            new_y = y+dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height and self.area[new_x][new_y] == 1:
                result.append((new_x,new_y))
        return result