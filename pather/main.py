def pather():
    from pather.classes.environment import Environment
    from env_parser import TOTAL_TIME,UAV_AMOUNT
    all_moves = {uav_index: [] for uav_index in range(UAV_AMOUNT)}
    env = Environment.get_instance()
    for _ in range(TOTAL_TIME):
        moves = env.iterate()
        for uav_index,move in moves.items():
            all_moves[uav_index].append(move)

    from pather.utils.utilities import save_to_output
    save_to_output(all_moves)
