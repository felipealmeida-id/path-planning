from enum import Enum
from coordObject import coordObject
from Obstacle import Obstacle


#################################
##########CONTEXT################
#################################
DIM = coordObject(15, 15)
BIGDIM = coordObject(30, 30)
ORIGIN = coordObject(0, 0)
UAVAMOUNT = 1
TIMELENGTH = 200
POIS = [coordObject(0.031, 0.909), coordObject(0.56, 0.09)]
POIS_TIMES = [10, 18, 18, 18]
OBSTACLES = [
    Obstacle(coordObject(0.94, 0.4), coordObject(0.95, 0.5), 1)
]

OBS_PUNISH = 0.9
# Time to charge must be aprox 2.5 times the BATTERY_CAPACITY
BATTERY_CAPACITY = 40
TIME_TO_CHARGE = 80


PAUSE_TIME = 0


class ACTION(Enum):
    STAY = 0
    RIGHT = 1
    DIAG_DOWN_RIGHT = 2
    DOWN = 3
    DIAG_DOWN_LEFT = 4
    LEFT = 5
    DIAG_UP_LEFT = 6
    UP = 7
    DIAG_UP_RIGHT = 8


colors = ['b', 'g', 'r', 'c', 'm', 'k']
markers = ['o', '^', 'v', '<', '>', 's',
           'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

metrics = ['Coverage', 'Collision', 'Obstacles', 'POIS', 'Uptime']
allMoves = [ACTION.STAY,
            ACTION.RIGHT,
            ACTION.DIAG_DOWN_RIGHT,
            ACTION.DOWN,
            ACTION.DIAG_DOWN_LEFT,
            ACTION.LEFT,
            ACTION.DIAG_UP_LEFT,
            ACTION.UP,
            ACTION.DIAG_UP_RIGHT
            ]
