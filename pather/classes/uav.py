from .coord import Coord
from enums import Move
from ..utils.constants import all_moves
from ..utils.utilities import distance,move_delta,delta_to_move
from env_parser import Env

class Uav:
    position:Coord
    battery:int
    current_charge:int
    max_battery:int
    charging:bool
    inbound_evaluation:float
    battery_evaluation:float

    def __init__(self,start:Coord,battery:int):
        self.position = start
        self.battery = battery
        self.max_battery = battery
        self.charging = False
        self.current_charge = 0
        self.inbound_evaluation = 1

    def get_effective_battery(self):
        return 0 if self.charging else (self.battery - self.distance_to_base())

    def move(self,move:Move):
        from .surveillance_area import SurveillanceArea
        env = Env.get_instance()
        environment = SurveillanceArea.get_instance()
        if self.charging:
            self.current_charge += 1
            if self.current_charge == env.UAV_CHARGE_TIME:
                self.charging = False
                self.current_charge = 0
                self.battery = self.max_battery
            return self.position,Move.STAY
        actual_move = move if self.get_effective_battery() > 0 else self.get_move_to_base()
        self.apply_move(actual_move)
        if (self.position == environment.start) and (self.get_effective_battery() < 2):
            self.charging = True
        return self.position,actual_move
    
    def apply_move(self,move:Move):
        delta = move_delta(move)
        self.position.apply_delta(delta)
        self.battery -= 1

    def evaluation_move(self,move:Move):
        from .surveillance_area import SurveillanceArea
        environment = SurveillanceArea.get_instance()
        env = Env.get_instance()
        self.apply_move(move)
        if not environment.is_inbound(self.position):
            self.inbound_evaluation -= 1/env.TOTAL_TIME
        return self.position

    def possible_moves(self):
        from .surveillance_area import SurveillanceArea
        env = SurveillanceArea.get_instance()
        end_positions = [(self.position.copy().apply_delta(move_delta(move)),move) for move in all_moves]
        filtered_moves = filter(lambda pos_move:env.is_inbound(pos_move[0]),end_positions)
        return list(map(lambda pos_move:pos_move[1],filtered_moves))
        
    def get_move_to_base(self):
        from .surveillance_area import SurveillanceArea
        env = SurveillanceArea.get_instance()
        delta = env.start - self.position
        return delta_to_move(delta)

    def distance_to_base(self):
        from .surveillance_area import SurveillanceArea
        env = SurveillanceArea.get_instance()
        return distance(self.position,env.start) + 1

    def __repr__(self):
        return f"""UAV
        position:{self.position} 
        battery:{self.battery}
        charging:{self.charging}
        current_charge:{self.current_charge}
        dist_to_base:{self.distance_to_base()}
        """