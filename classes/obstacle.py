from .coord import Coord

class Obstacle:
    sections:list[Coord]
    dims_init:Coord
    dims_end:Coord

    def __init__(self, dimsInit: Coord, dimsEnd: Coord):
        self.dimsInit = dimsInit
        self.dimsEnd = dimsEnd
        self.sections = self._to_sections()

    def _to_sections(self):
        from .environment import Environment
        sections = []
        env = Environment.get_instance()
        percentageOfSectionInitX = env.size.x * self.dimsInit.x
        percentageOfSectionEndX = env.size.x * self.dimsEnd.x
        percentageOfSectionInitY = env.size.y * self.dimsInit.y
        percentageOfSectionEndY = env.size.y * self.dimsEnd.y
        sectionInitX = int(percentageOfSectionInitX)
        sectionEndX = int(percentageOfSectionEndX)
        sectionInitY = int(percentageOfSectionInitY)
        sectionEndY = int(percentageOfSectionEndY)
        for i in range(sectionInitX, sectionEndX+1):
            for j in range(sectionInitY, sectionEndY+1):
                sections.append(Coord(i, j))
        return sections
