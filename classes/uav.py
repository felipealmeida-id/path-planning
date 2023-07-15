from classes.coord import Coord
from utils.enums import Move
from utils.constants import all_moves
from utils.utilities import distance,move_delta,delta_to_move

class Uav:
    position:Coord
    battery:int
    max_battery:int
    charging:bool 

    def __init__(self,start:Coord,battery:int):
        self.position = start
        self.battery = battery
        self.max_battery = battery
        self.charging = False

    def get_effective_battery(self):
        return 0 if self.charging else (self.battery - self.distance_to_base())

    def move(self,move:Move):
        from .environment import Environment
        env = Environment.get_instance()
        if self.charging:
            self.battery += 1
            if self.battery == self.max_battery:
                self.charging = False
            return
        delta = move_delta(move)
        self.position.apply_delta(delta)
        self.battery -= 1
        if (self.position == env.start) and (self.get_effective_battery() == 0):
            self.charging = True
    
    def possible_moves(self):
        from .environment import Environment
        env = Environment.get_instance()
        end_positions = [self.position.copy() for _ in all_moves]
        for pos,move in zip(end_positions,all_moves):
            pos.apply_delta(move_delta(move))
        filtered_moves = filter(env.is_inbound,end_positions)
        return list(map(delta_to_move,filtered_moves))
        
    def get_move_to_base(self):
        from .environment import Environment
        env = Environment.get_instance()
        delta = self.position - env.start
        return delta_to_move(delta)

    def distance_to_base(self):
        from .environment import Environment
        env = Environment.get_instance()
        return distance(self.position,env.start)