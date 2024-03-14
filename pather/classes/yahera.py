from random import choice
from .heuristic import MoveHeuristic
from .uav import Uav
from .point_of_interest import Point_Of_Interest
from enums import Move
from ..utils.constants import ardemisa_move_params
from ..utils.utilities import check_parameters,delta_to_move,move_delta
from ..utils.aStar import PathTraverser
from env_parser import Env
from .coord import Coord

class Yahera(MoveHeuristic):
    targeting:dict[int,Point_Of_Interest] = {}
    targetted_points:set[Point_Of_Interest] = set()
    visited:dict[(int,int),int] = {}

    def __init__(self) -> None:
        super().__init__()
        env = Env.get_instance()
        self.map = env.MAP
        self.traverser = PathTraverser(self.map)

    def get_move(self,**kwargs) -> Move:
        # Parse arguments
        check_parameters(kwargs,ardemisa_move_params)
        uav:Uav = kwargs.get('uav')
        uav_index:int = kwargs.get('uav_index')
        time:int = kwargs.get('time')
        points_of_interest:list[Point_Of_Interest] = kwargs.get("points_of_interest")
        # Mark the current position as visited
        self.visited[(
            uav.position.x, uav.position.y
        )] = self.visited.get((
            uav.position.x, uav.position.y
        ), 0) + 1
        
        print(f"-------------------UAV {uav_index} at time {time}-------------------")

        uav_current_target = self.choose_target(uav_index,time,points_of_interest)
        uav_possible_moves = uav.possible_moves()
        print("Im on:", uav.position.toTuple())
        print("Possible moves:", uav_possible_moves)
        uav_possible_moves = self._filterMoves(uav_possible_moves, uav.position)
        print("Filtered moves:", uav_possible_moves)
        if uav_current_target is None:
            selected_move = uav_possible_moves[0]
            return selected_move
        # if uav has target, move towards it
        print("I want to go to:", uav_current_target.position.toTuple())
        path = list(self.traverser.astar(uav.position.toTuple(), uav_current_target.position.toTuple()))
        next_point = Coord(path[1][0], path[1][1])

        uav_target_delta = next_point - uav.position
        chosen_move = delta_to_move(uav_target_delta)
        if uav.position.copy().apply_delta(move_delta(chosen_move)) == uav_current_target.position:
             self.targetted_points.remove(uav_current_target)
             self.targeting.pop(uav_index)
        print("I will move:", chosen_move)
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
    
    def _filterMoves(self, posibleMoves, current_pos:Coord):
        # Extraemos los movimientos que no estan sobre la trayectoria
        uav_possible_moves = [move for move in posibleMoves if self.map[current_pos.x + move_delta(move)[0]][current_pos.y + move_delta(move)[1]] == 1]
        # Ordenamos los movimientos por la cantidad de veces que se ha visitado el punto
        uav_possible_moves = sorted(uav_possible_moves, key=lambda move: self.visited.get((current_pos.x + move_delta(move)[0], current_pos.y + move_delta(move)[1])
            , 0))
        return uav_possible_moves

    
    def reset(self):
        self.targeting.clear()
        self.targetted_points.clear()