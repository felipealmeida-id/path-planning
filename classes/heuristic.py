from abc import abstractmethod
from classes import Uav
from utils import Move

class MoveHeuristic:
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self,uav:Uav) -> Move:
        pass