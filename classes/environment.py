from classes.obstacle import Obstacle
from classes.coord import Coord
from classes.uav import Uav
from classes.heuristic import MoveHeuristic
from classes.nefesto import Nefesto
from utils.env_parser import UAV_AMOUNT,ENVIRONMENT_X_AXIS,ENVIRONMENT_Y_AXIS,UAV_BATTERY

class Environment:
    __instance = None
    obstacles:list[Obstacle]
    uavs:list[Uav]
    heuristic:MoveHeuristic

    def __init__(self) -> None:
        self.size = Coord(ENVIRONMENT_X_AXIS,ENVIRONMENT_Y_AXIS)
        self.start = Coord(0,0)
        self.uavs = [Uav(self.start.copy(),UAV_BATTERY) for _ in range(UAV_AMOUNT)]
        self.obstacles = []
        self.pois = []
        self.heuristic = Nefesto()
    
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
        for uav in self.uavs:
            move = self.heuristic.get_move(uav)
            uav.move(move)
            print(uav.position)