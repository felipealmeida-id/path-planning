from enums import Move
from pather.classes.coord import Coord
from pather.classes.surveillance_area import SurveillanceArea

def populate_area(actions: list[list[Move]]) -> tuple[list[list[list[int]]], float, float, float]:
    from env_parser import Env
    env = Env.get_instance()
    area = SurveillanceArea.get_instance()
    num_uavs = len(actions)
    total_time = len(actions[0])

    # Pre-allocate memory for res using list comprehension
    res = [[[0] for _ in range(env.ENVIRONMENT_Y_AXIS)] for _ in range(env.ENVIRONMENT_X_AXIS)]

    # Initialize currentPos, oob_pen, uav_battery, ooBattery
    currentPos = [Coord(0, 0) for _ in range(num_uavs)]
    oob_pen = [0.0 for _ in range(num_uavs)]
    uav_battery = [float(env.UAV_BATTERY) for _ in range(num_uavs)]
    ooBattery = [1.0 for _ in range(num_uavs)]

    # Define a dictionary to map Move types to coordinate changes
    action_to_delta = {
        Move.RIGHT: (1, 0),
        Move.DIAG_DOWN_RIGHT: (1, -1),
        Move.DOWN: (0, -1),
        Move.DIAG_DOWN_LEFT: (-1, -1),
        Move.LEFT: (-1, 0),
        Move.DIAG_UP_LEFT: (-1, 1),
        Move.UP: (0, 1),
        Move.DIAG_UP_RIGHT: (1, 1)
    }

    time_oob = 0
    ooBatteryPenalization = 3 / env.UAV_BATTERY

    for uav in range(num_uavs):
        for time in range(total_time):
            oob_penalize = 0
            chosenMove = actions[uav][time]

            # Update coordinates using the dictionary
            if chosenMove in action_to_delta:
                dx, dy = action_to_delta[chosenMove]
                currentPos[uav].x += dx
                currentPos[uav].y += dy
            
            # If it is in the origin and the action is STAY, it charges the battery
            # We have to take into account that the battery charges from 0 to constants.BATTERY_CAPACITY in constants.TIME_TO_CHARGE
            if (
                currentPos[uav] == area.start
                and chosenMove == Move.STAY
                and uav_battery[uav] < env.UAV_BATTERY
            ):
                uav_battery[uav] += (env.UAV_BATTERY / env.UAV_CHARGE_TIME)
            # If it is not in the origin it uses battery
            elif (currentPos[uav] != area.start):
                uav_battery[uav] -= 1
            # If the battery is 0 or less, it is out of battery
            # we need to penalize it, we want to make it 0 if the battery became more negative than -constants.BATTERY_CAPACITY/3
            if uav_battery[uav] <= 0:
                if ooBattery[uav] > 0:
                    ooBattery[uav] -= ooBatteryPenalization
                    if ooBattery[uav] < 0:
                        ooBattery[uav] = 0

            how_far_x = min(currentPos[uav].x, env.ENVIRONMENT_X_AXIS - currentPos[uav].x - 1)
            how_far_y = min(currentPos[uav].y, env.ENVIRONMENT_Y_AXIS - currentPos[uav].y - 1)
            
            if how_far_x < 0 or how_far_y < 0:
                time_oob += 1
                oob_penalize += min(how_far_x, 0)
                oob_penalize += min(how_far_y, 0)
                oob_pen[uav] += oob_penalize
            else:
                # Compute indices once
                curr_x = int(currentPos[uav].x)
                curr_y = int(currentPos[uav].y)

                # in the 2d array (aka the area) we append the time in which the drone is in that position
                res[curr_x][curr_y].append(time)

    max_pen = max(oob_pen)
    max_pen = -max_pen / ((total_time + 1) * total_time)
    
    return (
        res,
        1 - max_pen,
        1 - time_oob / (num_uavs * total_time),
        sum(ooBattery) / num_uavs,
    )