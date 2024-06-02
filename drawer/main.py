import matplotlib.pyplot as plt
from enums import Move
from env_parser import Env
from pather.classes.coord import Coord
from pather.utils.utilities import move_delta


colors = ['b', 'g', 'r', 'c', 'm', 'k']

def draw_route(file:str, HR:bool = False):
    env = Env.get_instance()
    x_dim = env.HR_ENVIRONMENT_X_AXIS if HR else env.ENVIRONMENT_X_AXIS
    y_dim = env.HR_ENVIRONMENT_Y_AXIS if HR else env.ENVIRONMENT_Y_AXIS
    PAUSE_TIME = 0.1
    time = env.HR_TOTAL_TIME if HR else env.TOTAL_TIME
    _configure_graph(HR)
    _draw_pois(HR)
    _draw_obstacles(HR)
    route_list = _read_file(file)
    route_list_enum = list(enumerate(route_list))
    uav_pos = [Coord(env.START_X_COORD, env.START_Y_COORD) for _ in route_list]
    for t in range(time):
        for index, route in route_list_enum:
            if t >= len(route):
                continue
            move = route[t]
            delta = move_delta(move)
            move_start = uav_pos[index].copy()
            move_end = uav_pos[index].apply_delta(delta)
            if move == Move.STAY:
                _handle_consecutive_stays(t,index,route,move_end)
            plt.plot([move_start.x, move_end.x], [move_start.y, move_end.y], color=colors[index])
        timer = plt.annotate(str(t), [x_dim + 0.3, y_dim + 0.3], fontsize=22)
        dot_position = [plt.scatter(pos.x, pos.y, marker='o', color=colors[i], s=100) for i,pos in enumerate(uav_pos)]
        if(PAUSE_TIME > 0):
            plt.pause(PAUSE_TIME)
        _clear_position_dots(dot_position)
        timer.remove()
    timer = plt.annotate(str(t+1), [x_dim + 0.3, y_dim + 0.3], fontsize=22)
    plt.show()
#################################################################################################################################################################################################### 
#################################################################################################################################################################################################### 
#################################################################################################################################################################################################### 
def _configure_graph(HR):
    env = Env.get_instance()
    x_dim = env.HR_ENVIRONMENT_X_AXIS if HR else env.ENVIRONMENT_X_AXIS
    y_dim = env.HR_ENVIRONMENT_Y_AXIS if HR else env.ENVIRONMENT_Y_AXIS

    plt.figure()
    plt.xlim(-0.3, x_dim+0.3)
    plt.ylim(-0.3, y_dim+0.3)
    plt.grid(True)

    ticks = [t for t in range(x_dim)]
    plt.xticks(ticks=ticks)
    plt.yticks(ticks=ticks)

def _draw_pois(HR):
    env = Env.get_instance()
    POIs= env.HR_POINTS_OF_INTEREST_COORDS if HR else env.POINTS_OF_INTEREST_COORDS
    for x,y in POIs:
        plt.plot(x, y, marker='o', color='k')

def _draw_obstacles(HR):
    pass
    # for badSection in flatten_obstacles(dimensions):
    #     plt.fill([badSection.x, badSection.x+1, badSection.x+1, badSection.x],
    #              [badSection.y, badSection.y, badSection.y+1, badSection.y+1],
    #              color='black', alpha=0.5)

def _read_file(path:str):
    with open(path, 'r') as file:
        result = [[Move(int(x)) for x in line.strip().split()] for line in file]
    return result

def _handle_consecutive_stays(time:int,index:int,route:list[Move],move:Move):
    plt.scatter(move.x, move.y, s=80,facecolors='none', edgecolors=colors[index])
    if time == 0 or time not in range(len(route)) or route[time-1] != Move.STAY:
        acc = 1
        i = 1
        while time+i in range(len(route)) and route[time + i] == Move.STAY:
            acc += 1
            i += 1
        plt.annotate(acc, (move.x+0.1,move.y+0.1), color=colors[index])

def _clear_position_dots(dot_position):
    for dot in dot_position:
        dot.remove()
    dot_position.clear()