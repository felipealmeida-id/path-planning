class Coord:
    def __init__(self,x:int,y:int) -> None:
        self.x = x
        self.y = y
    
    def apply_delta(self,delta:tuple[int,int]):
        self.x = self.x+delta[0]
        self.y = self.y+delta[1]

    def copy(self):
        return Coord(self.x,self.y)
    
    def __eq__(self,other):
        return self.x == other.x and self.y == other.y