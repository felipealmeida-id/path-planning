from os import getenv
from classes import MoveHeuristic,Coord,Environment
from utils import move_delta,all_moves

class Uav:
    heuristic:MoveHeuristic
    position:Coord
    battery:int

    def __init__(self,heuristic:MoveHeuristic):
        env = Environment.get_instance()
        self.position = env.start
        self.heuristic = heuristic
        self.battery = getenv


    def move(self):
        chosen_move = self.heuristic.get_move(self)
        delta = move_delta(chosen_move)
        self.position.apply_delta(delta)
        self.battery -= 1
    
    def possible_moves(self):
        env = Environment.get_instance()
        end_positions = [self.position.copy() for _ in all_moves]
        for pos,move in zip(end_positions,all_moves):
            pos.apply_delta(move_delta(move))
        return list(filter(env.is_inbound,end_positions))
        