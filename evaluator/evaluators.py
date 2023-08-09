from pather.classes.coord import Coord
from enums import Move
from utilities import get_duplicates

def evaluate_coverage_area(area: list[list[list[int]]], _) -> float:
    from env_parser import Env
    env = Env.get_instance()
    numberOfSquares = env.HR_ENVIRONMENT_X_AXIS * env.HR_ENVIRONMENT_Y_AXIS
    res = numberOfSquares
    for i in range(env.HR_ENVIRONMENT_X_AXIS):
        for j in range(env.HR_ENVIRONMENT_Y_AXIS):
            if len(area[i][j]) == 0:
                res = res - 1
    return res / numberOfSquares


def evaluate_drones_collision(area: list[list[list[int]]], actions: list[list[Move]]) -> float:
    from env_parser import Env
    env = Env.get_instance()
    numberOfDrones = env.UAV_AMOUNT
    numberOfTimes = env.HR_TOTAL_TIME
    worstCase = numberOfDrones * numberOfTimes
    res = 0
    for i in range(env.HR_ENVIRONMENT_X_AXIS):
        for j in range(env.HR_ENVIRONMENT_Y_AXIS):
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


def evaluate_POI_coverage(area: list[list[list[int]]], actions: list[list[Move]]) -> float:
    from pather.classes.surveillance_area import SurveillanceArea
    from env_parser import Env
    env = Env.get_instance()
    surveillance_area = SurveillanceArea.get_instance()
    timeSpentNeedy = [0 for _ in surveillance_area.points_of_interest]
    lastVisit = [0 for _ in surveillance_area.points_of_interest]
    time = len(actions[0])
    pois = surveillance_area.points_of_interest
    for t in range(time):
        for i, poi in enumerate(pois):
            coords = poi.position
            x = coords.x
            y = coords.y
            if t in area[x][y]:  # type: ignore
                lastVisit[i] = t
            elif t - lastVisit[i] > env.HR_POINTS_OF_INTEREST_VISIT_TIMES[i]:
                timeSpentNeedy[i] += 1
    totalTimeSpentNeedy = 0
    for needy in timeSpentNeedy:
        totalTimeSpentNeedy += needy
    maxNeedyTimes = [time - poiTime for poiTime in env.HR_POINTS_OF_INTEREST_VISIT_TIMES]
    maximumNeediness = 0
    for needy in maxNeedyTimes:
        maximumNeediness += needy
    return 1 - totalTimeSpentNeedy / maximumNeediness

# Out of scope
def evaluate_drone_up_time(area: list[list[list[int]]], _: list[list['Move']]) -> float:
    from env_parser import Env
    env = Env.get_instance()
    dronesUp = 0

    # Directly check if there's a drone up at the start coordinates for each time t
    for t in range(env.HR_TOTAL_TIME):
        if t in area[env.START_X_COORD][env.START_Y_COORD]:
            dronesUp += 1

    return dronesUp / env.HR_TOTAL_TIME