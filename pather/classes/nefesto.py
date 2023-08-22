from random import choice
from .heuristic import MoveHeuristic
from .uav import Uav
from ..utils.constants import nefesto_move_params
from ..utils.utilities import check_parameters

class Nefesto(MoveHeuristic):
    def get_move(self,**kwargs):
        check_parameters(kwargs,nefesto_move_params)
        uav:Uav = kwargs.get('uav')

        uav_possible_moves = uav.possible_moves()
        return choice(uav_possible_moves)
    
    def reset(self):
        pass