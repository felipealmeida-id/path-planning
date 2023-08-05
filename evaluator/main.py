from enums import EvaluatorModules,Move
from evaluator.area_populator import populate_area

def evaluateGAN(generatedList: list[list[int]], activeModules: list[EvaluatorModules] = None):
    parsedList = _parseMoves(generatedList)
    return _evaluate(parsedList, activeModules)

def _evaluate(grid: list[list[Move]], activeModules: list[EvaluatorModules] | None = None):
    from evaluator.evaluators import evaluate_coverage_area,evaluate_POI_coverage,evaluate_drones_collision,evaluate_drone_up_time
    area, oob_dist, oob_time, batteryEvaluation = populate_area(grid)
    if activeModules is None:
        evaluators = {
            "Coverage": evaluate_coverage_area,
            "Collision": evaluate_drones_collision,
            "POIS": evaluate_POI_coverage,
            "Uptime": evaluate_drone_up_time,
            # "Obstacles": evaluateObstacles,
            "OutOfBound": lambda _, __: ((oob_dist + oob_time) / 2) ** 2,
        }
    else:
        evaluators = {}
        if EvaluatorModules.COVERAGE in activeModules:
            evaluators["Coverage"] = evaluate_coverage_area
        if EvaluatorModules.COLLISION in activeModules:
            evaluators["Collision"] = evaluate_drones_collision
        # if EvaluatorModules.OBSTACLES in activeModules:
            # evaluators["Obstacles"] = evaluateObstacles
        if EvaluatorModules.POIS in activeModules:
            evaluators["POIS"] = evaluate_POI_coverage
        # if EvaluatorModules.UPTIME in activeModules:
        #     evaluators["Uptime"] = evaluate_drone_up_time
        if EvaluatorModules.OUTOFBOUND in activeModules:
            evaluators["OutOfBound"] = (
                lambda _, __, ___: ((oob_dist + oob_time) / 2) ** 2
            )
        if EvaluatorModules.BATTERY in activeModules:
            evaluators["Battery"] = lambda _, __: batteryEvaluation
    evaluateMetric = lambda eval: eval(area, grid)
    results = {metric: evaluateMetric(eval) for metric, eval in evaluators.items()}
    accumulator = 0
    for v in results.values():
        accumulator += v
    return accumulator / len(results)

def _parseMoves(listOfLists: list[list[int]]) -> list[list[Move]]:
    return [list(map(Move,line)) for line in listOfLists]