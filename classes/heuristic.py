from abc import abstractmethod
from utils.enums import Move
from classes.uav import Uav

class MoveHeuristic:
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self,uav:Uav) -> Move:
        pass