from enums import Move
from utilities import get_duplicates

def evaluate_coverage_area(*args) -> float:
    from env_parser import Env

    area:list[list[list[int]]] = args[0]
    res = args[1]

    env = Env.get_instance()
    x_axis = env.HR_ENVIRONMENT_X_AXIS if res == "HIGH" else env.ENVIRONMENT_X_AXIS
    y_axis = env.HR_ENVIRONMENT_Y_AXIS if res == "HIGH" else env.ENVIRONMENT_Y_AXIS

    numberOfSquares = x_axis * y_axis
    res = numberOfSquares
    for i in range(x_axis):
        for j in range(y_axis):
            if len(area[i][j]) == 0:
                res = res - 1
    return res / numberOfSquares


def evaluate_drones_collision(*args) -> float:
    from env_parser import Env
    area:list[list[list[int]]] = args[0]
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

def evaluate_POI_coverage(*args) -> float:
    from pather.classes.surveillance_area import SurveillanceArea
    from env_parser import Env
    area: list[list[list[int]]] = args[0]
    actions: list[list[Move]] = args[1]
    res = args[2]
    env = Env.get_instance()

    poiTimeFromEnv = env.HR_POINTS_OF_INTEREST_VISIT_TIMES if res == "HIGH" else env.POINTS_OF_INTEREST_VISIT_TIMES
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
            elif t - lastVisit[i] > poiTimeFromEnv[i]:
                timeSpentNeedy[i] += 1
    totalTimeSpentNeedy = 0
    for needy in timeSpentNeedy:
        totalTimeSpentNeedy += needy
    maxNeedyTimes = [time - poiTime for poiTime in poiTimeFromEnv]
    maximumNeediness = 0
    for needy in maxNeedyTimes:
        maximumNeediness += needy
    return 1 - totalTimeSpentNeedy / maximumNeediness

# Out of scope
def evaluate_drone_up_time(*args) -> float:
    from env_parser import Env
    area: list[list[list[int]]] = args[0]
    env = Env.get_instance()
    dronesUp = 0
    for t in range(env.HR_TOTAL_TIME):
        if t in area[env.START_X_COORD][env.START_Y_COORD]:
            dronesUp += 1
    return dronesUp / env.HR_TOTAL_TIME

# Out of scope Probably
def evaluateObstacles(*args) -> float:
    area: list[list[list[int]]] = args[0]
    actions: list[list[Move]] = args[1]
    numberOfDrones = len(actions)
    numberOfTimes = len(actions[0])
    worstCase = numberOfDrones * numberOfTimes
    flat_obs = constants.FLAT_OBSTACLES
    timeOnObs = 0
    for obs in flat_obs:
        x = obs.x
        y = obs.y
        timeOnObs += len(area[x][y])
    return 1 - timeOnObs / worstCase