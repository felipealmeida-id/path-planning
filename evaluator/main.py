from enums import EvaluatorModules,Move
def evaluateGAN(generatedList: list[list[int]], activeModules: list[EvaluatorModules] = None):
    parsedList = parseMoves(generatedList)
    return evaluate(parsedList, activeModules)

def evaluate(grid: list[list[Move]], activeModules: list[EvaluatorModules] | None = None):
    from env_parser import Env
    from pather.classes import Coord
    env = Env.get_instance()
    gridDimensions = Coord(env.ENVIRONMENT_X_AXIS,env.ENVIRONMENT_Y_AXIS)
    area, oob_dist, oob_time, batteryEvaluation = populateArea(grid, gridDimensions)
    if activeModules is None:
        evaluators = {
            "Coverage": evaluateCoverageArea,
            "Collision": evaluateDronesCollision,
            # "Obstacles": evaluateObstacles,
            "POIS": evaluatePOICoverage,
            "Uptime": evaluateDroneUpTime,
            "OutOfBound": lambda _, __, ___: ((oob_dist + oob_time) / 2) ** 2,
        }
    else:
        evaluators = {}
        if EvaluatorModules.COVERAGE in activeModules:
            evaluators["Coverage"] = evaluateCoverageArea
        if EvaluatorModules.COLLISION in activeModules:
            evaluators["Collision"] = evaluateDronesCollision
        # if EvaluatorModules.OBSTACLES in activeModules:
            # evaluators["Obstacles"] = evaluateObstacles
        if EvaluatorModules.POIS in activeModules:
            evaluators["POIS"] = evaluatePOICoverage
        if EvaluatorModules.UPTIME in activeModules:
            evaluators["Uptime"] = evaluateDroneUpTime
        if EvaluatorModules.OUTOFBOUND in activeModules:
            evaluators["OutOfBound"] = (
                lambda _, __, ___: ((oob_dist + oob_time) / 2) ** 2
            )
        if EvaluatorModules.BATTERY in activeModules:
            evaluators["Battery"] = lambda _, __, ___: batteryEvaluation
    evaluateMetric = lambda eval: eval(area, grid, gridDimensions)
    results = {metric: evaluateMetric(eval) for metric, eval in evaluators.items()}
    accumulator = 0
    for v in results.values():
        accumulator += v
    return accumulator / len(results)

def parseMoves(listOfLists: list[list[int]]) -> list[list[Move]]:
    return [list(map(Move,line)) for line in listOfLists]