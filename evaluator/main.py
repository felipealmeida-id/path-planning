from enums import EvaluatorModules,Move
from evaluator.area_populator import populate_area
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import cpu_count

def evaluateGAN(generatedList: list[list[int]], activeModules: list[EvaluatorModules] = None):
    parsedList = _parseMoves(generatedList)
    return _evaluate(parsedList, activeModules)

def _evaluate(grid: list[list[Move]], active_modules: list[EvaluatorModules] | None = None):
    area, oob_dist, oob_time, battery_evaluation = populate_area(grid)
    constant_evals = {
        EvaluatorModules.OUTOFBOUND:((oob_dist + oob_time) / 2) ** 2,
        EvaluatorModules.BATTERY:battery_evaluation,
    }
    evaluators = _determine_evaluators(active_modules,constant_evals)
    results = {metric: evaluate(area,grid) for metric, evaluate in evaluators.items()}
    # results = _multi_thread_eval(evaluators,area,grid)
    accumulator = 0
    for v in results.values():
        accumulator += v
    return accumulator / len(results)

def _parseMoves(listOfLists: list[list[int]]) -> list[list[Move]]:
    return [list(map(Move,line)) for line in listOfLists]

def _determine_evaluators(active_modules,already_determined_evals):
    from evaluator.evaluators import evaluate_coverage_area,evaluate_POI_coverage,evaluate_drones_collision,evaluate_drone_up_time
    evaluators = {}
    for key,value in already_determined_evals.items():
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

def _multi_thread_eval(evaluators,area,grid):
    results = {}
    thread_num = cpu_count()
    with ThreadPoolExecutor(thread_num) as executor:
        futures = {executor.submit(func,area,grid): key for key, func in evaluators.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                results[key] = result
            except Exception as e:
                results[key] = e
    return results

def _constant_return_fun(return_value):
    return lambda *args : return_value