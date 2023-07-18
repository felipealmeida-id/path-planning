from random import choice
from .heuristic import MoveHeuristic
from .uav import Uav
from .point_of_interest import Point_Of_Interest
from utils.enums import Move
from utils.constants import ardemisa_move_params
from utils.utilities import check_parameters,delta_to_move

class Ardemisa(MoveHeuristic):
    targeting:dict[int,Point_Of_Interest]

    def get_move(self,**kwargs) -> Move:
        # Parse arguments
        check_parameters(kwargs,ardemisa_move_params)
        uav:Uav = kwargs.get('uav')
        uav_index:int = kwargs.get('uav_index')
        time:int = kwargs.get('time')
        points_of_interest:list[Point_Of_Interest] = kwargs.get("points_of_interest")

        # Choose target
        uav_current_target = self.targeting.get(uav_index)
        if uav_current_target is None:
            for poi in points_of_interest:
                if time - poi.last_visit > poi.visit_time:
                        self.targeting[uav_index] = poi

        uav_current_target = self.targeting.get(uav_index)
        uav_possible_moves = uav.possible_moves()
        if uav_current_target is None:
            return choice(uav_possible_moves)
        uav_target_delta = uav_current_target.position - uav.position
        return delta_to_move(uav_target_delta)
