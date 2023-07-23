from .coord import Coord
from ..utils.enums import Move
from ..utils.constants import all_moves
from ..utils.utilities import distance,move_delta,delta_to_move
from ..utils.env_parser import UAV_CHARGE_TIME

class Uav:
    position:Coord
    battery:int
    current_charge:int
    max_battery:int
    charging:bool 

    def __init__(self,start:Coord,battery:int):
        self.position = start
        self.battery = battery
        self.max_battery = battery
        self.charging = False
        self.current_charge = 0

    def get_effective_battery(self):
        return 0 if self.charging else (self.battery - self.distance_to_base())

    def move(self,move:Move):
        from .environment import Environment
        env = Environment.get_instance()
        if self.charging:
            self.current_charge += 1
            if self.current_charge == UAV_CHARGE_TIME:
                self.charging = False
                self.current_charge = 0
                self.battery = self.max_battery
            return self.position,Move.STAY
        actual_move = move if self.get_effective_battery() > 0 else self.get_move_to_base()
        delta = move_delta(actual_move)
        self.position.apply_delta(delta)
        self.battery -= 1
        if (self.position == env.start) and (self.get_effective_battery() < 2):
            self.charging = True
        return self.position,actual_move
    
    def possible_moves(self):
        from .environment import Environment
        env = Environment.get_instance()
        end_positions = [(self.position.copy().apply_delta(move_delta(move)),move) for move in all_moves]
        filtered_moves = filter(lambda pos_move:env.is_inbound(pos_move[0]),end_positions)
        return list(map(lambda pos_move:pos_move[1],filtered_moves))
        
    def get_move_to_base(self):
        from .environment import Environment
        env = Environment.get_instance()
        delta = env.start - self.position
        return delta_to_move(delta)

    def distance_to_base(self):
        from .environment import Environment
        env = Environment.get_instance()
        return distance(self.position,env.start) + 1

    def __repr__(self):
        return f"""UAV
        position:{self.position} 
        battery:{self.battery}
        charging:{self.charging}
        current_charge:{self.current_charge}
        dist_to_base:{self.distance_to_base()}
        """