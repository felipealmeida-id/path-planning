from abc import abstractmethod
from ..utils.enums import Move
from .uav import Uav

class MoveHeuristic:
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self,**kwargs) -> Move:
        pass