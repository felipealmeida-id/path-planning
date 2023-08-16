from ..classes.coord import Coord
from enums import Move
from env_parser import Env
from .constants import delta_dict

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

def move_delta(move:Move)->tuple[int,int]:
    return delta_dict[move]

def check_parameters(kwargs:dict[str,any],params:list[str]):
    for param in params:
        value_of_param = kwargs.get(param)
        if value_of_param == None:
            raise ValueError(f"Parameter {param} not provided to move heruistic")

def save_to_output(moves:dict[int,list[Move]],id:str):
    env = Env.get_instance()
    with open(f'./output/{env.PY_ENV}/pather/generated_paths/path.{id}.txt',"w") as file:
        lines_to_write = []
        for moves in moves.values():
            moves_as_nums = ' '.join(list(map(lambda move:str(move.value),moves)))
            lines_to_write.append(f"{moves_as_nums}\n")
        file.writelines(lines_to_write)