from pather.classes import Coord
from enums import Move

def evaluateCoverageArea(area: list[list[list[int]]], _) -> float:
    from env_parser import Env
    env = Env.get_instance()
    numberOfSquares = env.ENVIRONMENT_X_AXIS * env.ENVIRONMENT_Y_AXIS
    res = numberOfSquares
    for i in range(env.ENVIRONMENT_X_AXIS):
        for j in range(env.ENVIRONMENT_Y_AXIS):
            if len(area[i][j]) == 0:
                res = res - 1
    return res / numberOfSquares


def evaluateDronesCollision(area: list[list[list[int]]], actions: list[list[Move]]) -> float:
    from env_parser import Env
    env = Env.get_instance()
    numberOfDrones = env.UAV_AMOUNT
    numberOfTimes = env.TOTAL_TIME
    worstCase = numberOfDrones * numberOfTimes
    res = 0
    for i in range(env.ENVIRONMENT_X_AXIS):
        for j in range(env.ENVIRONMENT_Y_AXIS):
            if i == env.START_X_COORD and j == env.START_Y_COORD:
                continue
            duplicates = get_duplicates(area[i][j])
            for k in duplicates:
                res = res + duplicates[k]
    return 1 - (res / worstCase)


# def evaluateObstacles(area: list[list[list[int]]], actions: list[list[Move]], _) -> float:
#     numberOfDrones = len(actions)
#     numberOfTimes = len(actions[0])
#     worstCase = numberOfDrones * numberOfTimes
#     flat_obs = constants.FLAT_OBSTACLES
#     timeOnObs = 0
#     for obs in flat_obs:
#         x = obs.x
#         y = obs.y
#         timeOnObs += len(area[x][y])  # type: ignore
#     return 1 - timeOnObs / worstCase


def evaluatePOICoverage(area: list[list[list[int]]], actions: list[list[Move]]) -> float:
    timeSpentNeedy = [0 for _ in constants.POIS]
    lastVisit = [0 for _ in constants.POIS]
    time = len(actions[0])
    pois = [POI(coords, 0, 0) for coords in constants.POIS]
    for t in range(time):
        for i, poi in enumerate(pois):
            coords = poi.getSection(areaDims)
            x = coords.x
            y = coords.y
            if t in area[x][y]:  # type: ignore
                lastVisit[i] = t
            elif t - lastVisit[i] > constants.POIS_TIMES[i]:
                timeSpentNeedy[i] += 1
    totalTimeSpentNeedy = 0
    for needy in timeSpentNeedy:
        totalTimeSpentNeedy += needy
    maxNeedyTimes = [time - poiTime for poiTime in constants.POIS_TIMES]
    maximumNeediness = 0
    for needy in maxNeedyTimes:
        maximumNeediness += needy
    return 1 - totalTimeSpentNeedy / maximumNeediness


def evaluateDroneUpTime(
    area: list[list[list[int]]], actions: list[list[Move]], areaDims: Coord
) -> float:
    time = len(actions[0])
    dronesUp = 0
    breaked = False
    for t in range(time):
        for i in range(int(areaDims.x)):
            if breaked:
                breaked = False
                break
            for j in range(int(areaDims.y)):
                if i == constants.ORIGIN.x and j == constants.ORIGIN.y:
                    continue
                if t in area[i][j]:
                    dronesUp += 1
                    breaked = True
                    break
    return dronesUp / time