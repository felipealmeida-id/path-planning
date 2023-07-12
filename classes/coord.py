class Coord:
    def __init__(self,x:int,y:int) -> None:
        self.x = x
        self.y = y

    def apply_delta(self,delta:tuple[int,int]):
        self.x = self.x+delta[0]
        self.y = self.y+delta[1]

    def copy(self):
        return Coord(self.x,self.y)

    def __eq__(self,other:'Coord'):
        return self.x == other.x and self.y == other.y
    
    def __add__(self,other:'Coord'):
        return Coord(self.x+other.x,self.y+other.y)

    def __sub__(self,other:'Coord'):
        return Coord(self.x-other.x,self.y-other.y)

    def __neg__(self):
        return Coord(-self.x,-self.y)
    
    def __str__(self):
        return f"({self.x},{self.y})"
