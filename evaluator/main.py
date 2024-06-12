from enums import EvaluatorModules, Move
from evaluator.area_populator import populate_area, populate_area_v2
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import json


def evaluateGAN(
    generatedList: list[list[int]]):
    # parsedList = _parseMoves(generatedList)
    return _evaluate(generatedList)

def _weighted_evaluate(
        coverage,
        poiEval,
        oob_dist,
        oob_time,
        cohesion_evaluation):
       # Define weights
    weight_coverage = 1
    weight_poiEval = 2  # Increase the weight of poiEval
    weight_oob_dist = 1
    weight_oob_time = 1
    weight_cohesion = 1

    # Calculate the weighted sum
    weighted_sum = (
        weight_coverage * coverage +
        weight_poiEval * poiEval +
        weight_oob_dist * oob_dist +
        weight_oob_time * oob_time +
        weight_cohesion * cohesion_evaluation
    )

    # Calculate the sum of weights
    total_weights = (
        weight_coverage +
        weight_poiEval +
        weight_oob_dist +
        weight_oob_time +
        weight_cohesion
    )

    # Normalize the weighted sum
    final_score = weighted_sum / total_weights

    return final_score


def _evaluate(
    grid: list[list[Move]]):
    from evaluator.evaluators import (
        evaluate_coverage_area,
        evaluate_POI_coverage,
    )
    # to Move
    grid = _parseMoves(grid)
    res = "LOW"
    area, oob_dist, oob_time, battery_evaluation = populate_area(grid, res)
    coverage = evaluate_coverage_area(area, res)
    poiEval = evaluate_POI_coverage(area, grid, res)
    # we set cohesion to 100 because when using actions there are no jumps
    cohesion_evaluation = 1.0

    return _weighted_evaluate(coverage, poiEval, oob_dist, oob_time, cohesion_evaluation)


def _parseMoves(listOfLists: list[list[int]]) -> list[list[Move]]:
    return [list(map(Move, line)) for line in listOfLists]


def _determine_evaluators(active_modules, already_determined_evals):
    from evaluator.evaluators import (
        evaluate_coverage_area,
        evaluate_POI_coverage,
        evaluate_drones_collision,
        evaluate_drone_up_time,
    )

    evaluators = {}
    for key, value in already_determined_evals.items():
        if active_modules is None or key in active_modules:
            evaluators[key] = _constant_return_fun(value)
    evaluator_mapping = {
        EvaluatorModules.COVERAGE: ("Coverage", evaluate_coverage_area),
        EvaluatorModules.COLLISION: ("Collision", evaluate_drones_collision),
        EvaluatorModules.POIS: ("POIS", evaluate_POI_coverage),
        # EvaluatorModules.OBSTACLES: ("Obstacles", evaluateObstacles),
        # EvaluatorModules.UPTIME: ("Uptime", evaluate_drone_up_time),
    }
    for module, (evaluator_key, evaluator_func) in evaluator_mapping.items():
        if active_modules is None or module in active_modules:
            evaluators[evaluator_key] = evaluator_func
    return evaluators


def _multi_thread_eval(evaluators, area, grid):
    results = {}
    thread_num = cpu_count()
    with ThreadPoolExecutor(thread_num) as executor:
        futures = {
            executor.submit(func, area, grid): key for key, func in evaluators.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                results[key] = result
            except Exception as e:
                results[key] = e
    return results


def _constant_return_fun(return_value):
    return lambda *args: return_value

def evaluate_actions(path, res="LOW"):
    from evaluator.evaluators import (
        evaluate_coverage_area,
        evaluate_POI_coverage,
    )
    action_list: list[list[Move]]
    # file contains various line, eachline should be a list of moves which is an enum
    with open(path, "r") as f:
        action_list = [[Move(int(x)) for x in line.strip().split()] for line in f]
    area, oob_dist, oob_time, battery_evaluation = populate_area(action_list, res)
    coverage = evaluate_coverage_area(area, res)
    poiEval = evaluate_POI_coverage(area, action_list, res)
    # we set cohesion to 1 because when using actions there are no jumps
    cohesion_evaluation = 1.0

    return _weighted_evaluate(coverage, poiEval, oob_dist, oob_time, cohesion_evaluation), coverage, poiEval, oob_dist, oob_time, cohesion_evaluation
    
def evaluate_cartesian(path, res):
    grid: list[list[list[int]]]
    with open(path, "r") as f:
        grid = json.load(f)
    return _evaluate_cartesian(grid,res)
   

def _evaluate_cartesian(grid: list[list[list[int]]],res):
    from evaluator.evaluators import (
        evaluate_coverage_area,
        evaluate_POI_coverage,
    )
    area, oob_dist, oob_time, battery_evaluation, cohesion_evaluation = populate_area_v2(grid, res)
    coverage = evaluate_coverage_area(area, res)
    # Because evaluate pois does len of this and beacuse it is not a list of moves but cartesian reprsentation
    grid.pop()
    poiEval = evaluate_POI_coverage(area, grid, res)

    return _weighted_evaluate(coverage, poiEval, oob_dist, oob_time, cohesion_evaluation), coverage, poiEval, oob_dist, oob_time, cohesion_evaluation