from random import choice
from .heuristic import MoveHeuristic
from .uav import Uav
from .point_of_interest import Point_Of_Interest
from enums import Move
from ..utils.constants import ardemisa_move_params
from ..utils.utilities import check_parameters,delta_to_move,move_delta
from env_parser import Env
from .coord import Coord

class Yahera(MoveHeuristic):
    targeting:dict[int,Point_Of_Interest] = {}
    targetted_points:set[Point_Of_Interest] = set()
    map:list[list[int]]
    dead_ends = set()
    last_not_deadend_move_index = -1

    def __init__(self) -> None:
        super().__init__()
        env = Env.get_instance()
        self.map = env.MAP
        self.visited = {}  # Tracks visited points and visit counts


    def get_move(self,**kwargs) -> Move:
        # Parse arguments
        check_parameters(kwargs,ardemisa_move_params)
        uav:Uav = kwargs.get('uav')
        uav_index:int = kwargs.get('uav_index')
        time:int = kwargs.get('time')
        points_of_interest:list[Point_Of_Interest] = kwargs.get("points_of_interest")
        moves = kwargs.get("moves")

        uav_current_target = self.choose_target(uav_index,time,points_of_interest)
        uav_possible_moves = uav.possible_moves()

        # Update visited count
        current_pos = (uav.position.x, uav.position.y)
        self.visited[current_pos] = self.visited.get(current_pos, 0) + 1

        # This heuristic is the same as Ardemisa's, but will only travel using the path that is set
        # The path is a 2d array of 0s and 1s, where 0 is a wall and 1 is a path
        curr_pos = uav.position
        uav_possible_moves = self._filterMoves(uav_possible_moves, curr_pos)
        print(f"-------------------UAV {uav_index} at time {time}-------------------")
        print("Im on:", curr_pos)
        print("I can make this moves:",uav_possible_moves)
        print("Dead ends:", self.dead_ends)
        
        # If i cant move, return None, this will finish the execution
        if(len(uav_possible_moves) == 0):
            return None
        
        if(len(uav_possible_moves) == 1) and self._isOpposite(uav_possible_moves[0], 
                                                              moves[self.last_not_deadend_move_index]):
            self.dead_ends.add(curr_pos.copy())
            self.last_not_deadend_move_index -= 1
            return uav_possible_moves[0]
        
        # Reset the index beacuse we are not in a dead end
        self.last_not_deadend_move_index = -1
       
        chosen_move = uav_possible_moves[0]
        if uav_current_target is None:
            return chosen_move
        else:
            uav_target_delta = uav_current_target.position - uav.position
            move_to_target = delta_to_move(uav_target_delta)
            move_decomposed = self._decomposeMove(move_to_target)
            # checks if there is a move in decomposed move that is in uav_possible_moves
            for move in uav_possible_moves:
                if move in move_decomposed:
                    chosen_move = move
                    break
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

    
    def _filterMoves(self, posibleMoves, current_pos:Coord):
        # Extraemos los movimientos que no estan sobre la trayectoria
        uav_possible_moves = [move for move in posibleMoves if self.map[current_pos.x + move_delta(move)[0]][current_pos.y + move_delta(move)[1]] == 1]
        # Quitamos los movimientos que nos llevan a un punto muerto
        uav_possible_moves = [move for move in uav_possible_moves if current_pos.copy().apply_delta(move_delta(move)) not in self.dead_ends]
        # # Filter out moves that have been visited more than 3 times only if there are other moves available
        # if len(uav_possible_moves) > 2:
        #     uav_possible_moves = [move for move in uav_possible_moves if self.visited.get((current_pos.x + move_delta(move)[0], current_pos.y + move_delta(move)[1]), 0) < 3]
        # Sort moves by less visited first
        uav_possible_moves = sorted(uav_possible_moves, key=lambda move: self.visited.get((current_pos.x + move_delta(move)[0], current_pos.y + move_delta(move)[1]), 0))
        return uav_possible_moves

    # Movimiento opuesto
    def _isOpposite(self, move1, move2):
        oposite = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
            Move.DIAG_DOWN_LEFT: Move.DIAG_UP_RIGHT,
            Move.DIAG_DOWN_RIGHT: Move.DIAG_UP_LEFT,
            Move.DIAG_UP_LEFT: Move.DIAG_DOWN_RIGHT,
            Move.DIAG_UP_RIGHT: Move.DIAG_DOWN_LEFT
        }
        return oposite[move1] == move2
    def _decomposeMove(self, move):
        if move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            return [move]
        else:
            switcher = {
                Move.DIAG_DOWN_LEFT: [Move.DOWN, Move.LEFT],
                Move.DIAG_DOWN_RIGHT: [Move.DOWN, Move.RIGHT],
                Move.DIAG_UP_LEFT: [Move.UP, Move.LEFT],
                Move.DIAG_UP_RIGHT: [Move.UP, Move.RIGHT]
            }
            # return the switcher option plus the move itself in an array
            res = switcher.get(move)
            res.append(move)
            return res