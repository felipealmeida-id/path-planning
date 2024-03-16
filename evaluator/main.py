from enums import EvaluatorModules, Move
from evaluator.area_populator import populate_area, populate_area_v2
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import json

def evaluateGAN(
    generatedList: list[list[list[int]]], activeModules: list[EvaluatorModules] = None
):
    # parsedList = _parseMoves(generatedList)
    return _evaluate(generatedList, activeModules)


def _evaluate(
    grid: list[list[list[int]]], active_modules: list[EvaluatorModules] | None = None
):
    area, oob_dist, oob_time, battery_evaluation, cohesion_evaluation = populate_area_v2(grid)
    constant_evals = {
        EvaluatorModules.OUTOFBOUND: ((oob_dist + oob_time) / 2),
        EvaluatorModules.BATTERY: battery_evaluation,
        EvaluatorModules.COHESION: cohesion_evaluation,
    }
    evaluators = _determine_evaluators(active_modules, constant_evals)
    results = {metric: evaluate(area, grid) for metric, evaluate in evaluators.items()}
    accumulator = 0
    for v in results.values():
        accumulator += v
    return accumulator / len(results)


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

def evaluate_actions(path, res):
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

    return (coverage + poiEval) / 2, coverage, poiEval
    
def evaluate_cartesian(path, res):
    from evaluator.evaluators import (
        evaluate_coverage_area,
        evaluate_POI_coverage,
    )
    grid: list[list[list[int]]]
    with open(path, "r") as f:
        grid = json.load(f)
    area, oob_dist, oob_time, battery_evaluation, cohesion_evaluation = populate_area_v2(grid, res)
    coverage = evaluate_coverage_area(area, res)
    # Because evaluate pois does len of this and beacuse it is not a list of moves but cartesian reprsentation
    grid.pop()
    poiEval = evaluate_POI_coverage(area, grid, res)

    return (coverage + poiEval) / 2, coverage, poiEval
    