from classes.coord import Coord
from utils.enums import Move

def delta_to_move(coord:Coord) -> Move:
    if coord.x > 0:
        if coord.y > 0:
            return Move.DIAG_UP_RIGHT
        elif coord.y < 0:
            return Move.DIAG_DOWN_RIGHT
        return Move.RIGHT
    
    elif coord.x < 0:
        if coord.y > 0:
            return Move.DIAG_UP_LEFT
        elif coord.y < 0:
            return Move.DIAG_DOWN_LEFT
        return Move.LEFT
    
    if coord.y > 0:
            return Move.UP
    elif coord.y < 0:
            return Move.DOWN
    return Move.STAY

def distance(one:Coord,other:Coord):
    """Octile distance"""
    diff = other - one
    distance = 0
    while diff != Coord(0,0):
        move = delta_to_move(-diff)
        delta = move_delta(move)
        diff.apply_delta(delta)
        distance += 1
    return distance

def x_delta(move: Move):
    """
    Calculates the shift a Move produces in the X axis
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
    Calculates the shift a Move produces in the Y axis
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
