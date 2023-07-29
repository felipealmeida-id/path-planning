from abc import abstractmethod
from enums import Move

class MoveHeuristic:
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self,**kwargs) -> Move:
        pass

    @abstractmethod
    def reset(self):
        pass