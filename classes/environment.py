from classes.obstacle import Obstacle
from classes.coord import Coord
from classes.uav import Uav

class Environment:
    __instance = None
    obstacles:list[Obstacle]
    uavs:list[Uav]

    def __init__(self) -> None:
        self.size = Coord(0,0)
        self.obstacles = []
        self.pois = []
        self.start = Coord(0,0)
    
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