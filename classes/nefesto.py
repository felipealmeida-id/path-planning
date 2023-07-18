from .heuristic import MoveHeuristic
from .uav import Uav
from .point_of_interest import Point_Of_Interest
from utils.constants import nefesto_move_params
from utils.utilities import check_parameters

class Nefesto(MoveHeuristic):
    def get_move(self,**kwargs):
        check_parameters(kwargs,nefesto_move_params)
        uav:Uav = kwargs.get('uav')
        uav_index:int = kwargs.get('uav_index')
        time:int = kwargs.get('time')
        points_of_interest:list[Point_Of_Interest] = kwargs.get('points_of_interest')