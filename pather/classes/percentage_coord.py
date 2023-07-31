from pather.classes.coord import Coord
class PercentageCoord:
    def __init__(self,x:int,y:int) -> None:
        is_percent = lambda num: num >= 0 or num <= 1
        if not is_percent(x) or not is_percent(y):
            raise ValueError(f"x={x} or y={y} is not a valid percent")
        self.x = x
        self.y = y
    
    def to_coord(self) -> Coord:
        from pather.classes.surveillance_area import SurveillanceArea
        environment = SurveillanceArea.get_instance()
        x = round(self.x * environment.size.x)
        y = round(self.y * environment.size.y)
        return Coord(x,y)
    
    def __repr__(self):
        return f"({self.x}%,{self.y}%)"