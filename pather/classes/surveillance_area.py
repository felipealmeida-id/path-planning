from .obstacle import Obstacle
from .coord import Coord
from .uav import Uav
from .heuristic import MoveHeuristic
from .ardemisa import Ardemisa
from .nefesto import Nefesto
from .point_of_interest import Point_Of_Interest
from enums import Move
from env_parser import Env

class SurveillanceArea:
    __instance = None
    obstacles:list[Obstacle]
    uavs:list[Uav]
    points_of_interest:list[Point_Of_Interest]
    heuristic:MoveHeuristic
    total_time:int
    time_elapsed:int

    def __init__(self) -> None:
        env = Env.get_instance()
        self.size = Coord(env.ENVIRONMENT_X_AXIS,env.ENVIRONMENT_Y_AXIS)
        self.start = Coord(env.START_X_COORD,env.START_Y_COORD)
        self.uavs = [Uav(self.start.copy(),env.UAV_BATTERY) for _ in range(env.UAV_AMOUNT)]
        self.obstacles = [Obstacle(Coord(x,y)) for (x,y) in env.OBSTACLES_COORDS]
        pois_times_and_coords = zip(env.POINTS_OF_INTEREST_VISIT_TIMES,env.POINTS_OF_INTEREST_COORDS)
        self.points_of_interest = [Point_Of_Interest(visit_time,Coord(x,y)) for visit_time,(x,y) in pois_times_and_coords]
        self.heuristic = Ardemisa()
        self.total_time = env.TOTAL_TIME
        self.time_elapsed = 0
    
    @classmethod
    def get_instance(cls):
        if not cls.__instance:
            cls.__instance = SurveillanceArea()
        return cls.__instance
    
    def is_inbound(self,coord:Coord):
        x_inbound = coord.x >= 0 and coord.x < self.size.x
        y_inbound = coord.y >= 0 and coord.y < self.size.y
        return x_inbound and y_inbound
    
    def obstacle_collide(self,coord:Coord) -> bool:
        collisions = [coord in obs.sections for obs in self.obstacles]
        return any(collisions)
    
    def iterate(self):
        resulting_moves:dict[int,Move] = {}
        surveyed_coords = []
        enumeration_uavs = enumerate(self.uavs)
        for uav_index,uav in enumeration_uavs:
            if uav.charging:
                move = Move.STAY
            else:
                move = self.heuristic.get_move(uav=uav,time=self.time_elapsed,uav_index=uav_index,points_of_interest=self.points_of_interest)
            surveying,performed_move = uav.move(move)
            surveyed_coords.append(surveying)
            resulting_moves[uav_index] = performed_move
        for poi in self.points_of_interest:
            if poi.position in surveyed_coords:
                poi.visit(self.time_elapsed)
        self.time_elapsed += 1
        return resulting_moves
    
    def reset(self):
        self.time_elapsed = 0
        self.heuristic.reset()
        for poi in self.points_of_interest:
            poi.last_visit = 0
        for uav in self.uavs:
            uav.battery = uav.max_battery
            uav.position = self.start.copy()
            uav.charging = False
            uav.current_charge = 0