from classes.coord import Coord
from utils.enums import Move
from utils.constants import all_moves
from utils.utilities import distance,move_delta

class Uav:
    position:Coord
    battery:int

    def __init__(self):
        from classes import Environment
        env = Environment.get_instance()
        self.position = env.start.copy()
        # self.battery = getenv

    def effective_battery(self):
        return self.battery - self.distance_to_base()

    def move(self,move:Move):
        delta = move_delta(move)
        self.position.apply_delta(delta)
        self.battery -= 1
    
    def possible_moves(self):
        from classes import Environment
        env = Environment.get_instance()
        end_positions = [self.position.copy() for _ in all_moves]
        for pos,move in zip(end_positions,all_moves):
            pos.apply_delta(move_delta(move))
        return list(filter(env.is_inbound,end_positions))
        
    def move_to_base(self):
        from .environment import Environment
        env = Environment.get_instance()
        delta = self.position - env.start
        self.position.apply_delta(delta)
        self.battery -= 1

    def distance_to_base(self):
        from classes.environment import Environment
        env = Environment.get_instance()
        return distance(self.position,env.start)