from .coord import Coord

class Point_Of_Interest:
    visit_time:int
    last_visit:int
    position:Coord

    def __init__(self,visit_time:int,position:Coord):
        self.last_visit = 0
        self.visit_time = visit_time
        self.position = position

    def visit(self,time:int):
        self.last_visit = time
