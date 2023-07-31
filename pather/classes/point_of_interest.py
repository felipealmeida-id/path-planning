from .coord import Coord

class Point_Of_Interest:
    visit_time:int
    last_visit:int
    position:Coord
    visit_evaluation:float

    def visit(self,time:int):
        self.last_visit = time
        self.visit_evaluation = 1

    def evaluation_iteration(self,time:int) -> None:
        from env_parser import Env
        env = Env.get_instance()
        if time - self.last_visit > self.visit_time:
            self.visit_evaluation -= 1/env.TOTAL_TIME

    def __init__(self,visit_time:int,position:Coord):
        self.last_visit = 0
        self.visit_time = visit_time
        self.position = position

    def __repr__(self):
        return f"POI at {self.position}"

    def __eq__(self,other:'Point_Of_Interest'):
        return self.position == other.position
    
    def __hash__(self):
        return hash((self.position.x,self.position.y,self.visit_time))