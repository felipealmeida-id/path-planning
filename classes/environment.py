from classes.obstacle import Obstacle
from classes.coord import Coord
from classes.uav import Uav
from classes.heuristic import MoveHeuristic
from classes.nefesto import Nefesto
from classes.point_of_interest import Point_Of_Interest
from utils.env_parser import UAV_AMOUNT,ENVIRONMENT_X_AXIS,ENVIRONMENT_Y_AXIS,UAV_BATTERY,TOTAL_TIME,POINTS_OF_INTEREST_AMOUNT

class Environment:
    __instance = None
    obstacles:list[Obstacle]
    uavs:list[Uav]
    heuristic:MoveHeuristic
    total_time:int
    time_elapsed:int
    points_of_interest:list[Point_Of_Interest]

    def __init__(self) -> None:
        self.size = Coord(ENVIRONMENT_X_AXIS,ENVIRONMENT_Y_AXIS)
        self.start = Coord(0,0)
        self.uavs = [Uav(self.start.copy(),UAV_BATTERY) for _ in range(UAV_AMOUNT)]
        self.obstacles = []
        self.points_of_interest = [Point_Of_Interest for _ in range(POINTS_OF_INTEREST_AMOUNT)]
        self.heuristic = Nefesto()
        self.total_time = TOTAL_TIME
        self.time_elapsed = 0
    
    @classmethod
    def get_instance(cls):
        if not cls.__instance:
            cls.__instance = Environment()
        return cls.__instance
    
    def is_inbound(self,coord:Coord):
        x_inbound = coord.x >= 0 and coord.x < self.size.x
        y_inbound = coord.y >= 0 and coord.y < self.size.y
        return x_inbound and y_inbound
    
    def obstacle_collide(self,coord:Coord) -> bool:
        collisions = [coord in obs.sections for obs in self.obstacles]
        return any(collisions)
    
    def iterate(self):
        surveyed_coords = []
        enumeration_uavs = enumerate(self.uavs)
        for uav_index,uav in enumeration_uavs:
            move = self.heuristic.get_move(uav=uav,time=self.time_elapsed,uav_index=uav_index)
            surveying = uav.move(move)
            surveyed_coords.append(surveying)
        self.time_elapsed += 1