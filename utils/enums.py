from enum import Enum

class Move(Enum):
    STAY = 0
    RIGHT = 1
    DIAG_DOWN_RIGHT = 2
    DOWN = 3
    DIAG_DOWN_LEFT = 4
    LEFT = 5
    DIAG_UP_LEFT = 6
    UP = 7
    DIAG_UP_RIGHT = 8

def x_delta(move: Move):
    """
    Calculates the shift an Move produces in the X axis
    """
    positives = [Move.DIAG_DOWN_RIGHT, Move.RIGHT, Move.DIAG_UP_RIGHT]
    negatives = [Move.DIAG_DOWN_LEFT, Move.LEFT, Move.DIAG_UP_LEFT]
    if move in positives:
        return 1
    elif move in negatives:
        return -1
    else:
        return 0

def y_delta(move: Move):
    """
    Calculates the shift an Move produces in the Y axis
    """
    positives = [Move.DIAG_UP_RIGHT, Move.DIAG_UP_LEFT, Move.UP]
    negatives = [Move.DIAG_DOWN_RIGHT, Move.DIAG_DOWN_LEFT, Move.DOWN]
    if move in positives:
        return 1
    elif move in negatives:
        return -1
    else:
        return 0

def move_delta(move:Move):
    dx = x_delta(move)
    dy = y_delta(move)
    return dx , dy
