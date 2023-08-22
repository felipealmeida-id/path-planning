def pather():
    from pather.utils.utilities import save_to_output
    from pather.classes.surveillance_area import SurveillanceArea
    from env_parser import Env
    env = Env.get_instance()
    environment = SurveillanceArea.get_instance()
    for id in range(env.SAMPLES_TO_GENERATE):
        environment.reset()
        all_moves = {uav_index: [] for uav_index in range(env.UAV_AMOUNT)}
        for _ in range(env.TOTAL_TIME):
            moves = environment.iterate()
            for uav_index,move in moves.items():
                all_moves[uav_index].append(move)
        save_to_output(all_moves,id)
