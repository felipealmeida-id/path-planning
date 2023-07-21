try:
    # Initialize specific environment
    import sys
    from os import environ
    environ['PY_ENV'] = sys.argv[1]

    from random import seed
    from classes.environment import Environment
    from utils.env_parser import TOTAL_TIME,UAV_AMOUNT

    seed(2023)
    all_moves = {uav_index: [] for uav_index in range(UAV_AMOUNT)}
    env = Environment.get_instance()
    for i in range(TOTAL_TIME):
        moves = env.iterate()
        for uav_index,move in moves.items():
            all_moves[uav_index].append(move)
    
    from utils.utilities import save_to_output
    save_to_output(all_moves)
except Exception as e:
    print(str(e.args[0]))