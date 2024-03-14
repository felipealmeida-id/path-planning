def pather():
    from pather.utils.utilities import save_to_output
    from pather.classes.surveillance_area import SurveillanceArea
    from env_parser import Env
    env = Env.get_instance()
    environment = SurveillanceArea.get_instance(aStar=True)
    for id in range(env.SAMPLES_TO_GENERATE):
        environment.reset()
        all_moves = {uav_index: [] for uav_index in range(env.UAV_AMOUNT)}
        move = {
            0:1 
        }

        time = 0
        while move[0] is not None and time < env.TOTAL_TIME:
            move = environment.iterate(all_moves[0])
            if(move[0] is not None):
                all_moves[0].append(move[0])
            time += 1
        save_to_output(all_moves,id)
