from enum import Enum

class Move(Enum):
    STAY = 0
    RIGHT = 1
    DIAG_DOWN_RIGHT = 2
    DOWN = 3
    DIAG_DOWN_LEFT = 4
    LEFT = 5
    DIAG_UP_LEFT = 6
    UP = 7
    DIAG_UP_RIGHT = 8
    
    def __repr__(self):
        # return self.name
        if self.name == 'STAY':
            return 'o'
        if self.name == 'RIGHT':
            return '⇒'
        if self.name == 'DIAG_DOWN_RIGHT':
            return '⇘'
        if self.name == 'DOWN':
            return '⇓'
        if self.name == 'DIAG_DOWN_LEFT':
            return '⇙'
        if self.name == 'LEFT':
            return '⇐'
        if self.name == 'DIAG_UP_LEFT':
            return '⇖'
        if self.name == 'UP':
            return '⇑'
        if self.name == 'DIAG_UP_RIGHT':
            return '⇗'
        raise ValueError(f"{self.name} is invalid Move")
