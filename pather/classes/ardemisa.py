from random import choice
from .heuristic import MoveHeuristic
from .uav import Uav
from .point_of_interest import Point_Of_Interest
from enums import Move
from ..utils.constants import ardemisa_move_params
from ..utils.utilities import check_parameters,delta_to_move,move_delta

class Ardemisa(MoveHeuristic):
    targeting:dict[int,Point_Of_Interest] = {}
    targetted_points:set[Point_Of_Interest] = set()

    def get_move(self,**kwargs) -> Move:
        # Parse arguments
        check_parameters(kwargs,ardemisa_move_params)
        uav:Uav = kwargs.get('uav')
        uav_index:int = kwargs.get('uav_index')
        time:int = kwargs.get('time')
        points_of_interest:list[Point_Of_Interest] = kwargs.get("points_of_interest")

        uav_current_target = self.choose_target(uav_index,time,points_of_interest)
        uav_possible_moves = uav.possible_moves()
        if uav_current_target is None:
            return choice(uav_possible_moves)
        uav_target_delta = uav_current_target.position - uav.position
        chosen_move = delta_to_move(uav_target_delta)
        if uav.position.copy().apply_delta(move_delta(chosen_move)) == uav_current_target.position:
             self.targetted_points.remove(uav_current_target)
             self.targeting.pop(uav_index)
        return chosen_move

    def choose_target(self,uav_index:int,time:int,points_of_interest:list[Point_Of_Interest]):
        "Sets the uav's target in the targetting dictionary and returns the target"
        uav_current_target = self.targeting.get(uav_index)
        if uav_current_target is None:
            for poi in points_of_interest:
                if poi not in self.targetted_points and (time - poi.last_visit > poi.visit_time):
                    self.targeting[uav_index] = poi
                    self.targetted_points.add(poi)
                    return poi
        return uav_current_target
    
    def reset(self):
        self.targeting.clear()
        self.targetted_points.clear()