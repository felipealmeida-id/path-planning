import os
from json import dumps as json_dump
from tqdm import tqdm
move_encode_map = {
    0:[0,0],
    1:[1,0],
    2:[1,-1],
    3:[0,-1],
    4:[-1,-1],
    5:[-1,0],
    6:[-1,1],
    7:[0,1],
    8:[1,1]
}

def move_to_delta(route_set):
    res = []
    for uav_route in route_set:
        res_uav_route = []
        for move in uav_route:
            res_uav_route.append(move_encode_map[move])
        res.append(res_uav_route)
    return res

def delta_to_cartesian(route_set):
    res = []
    for uav_route in route_set:
        curr_position = [0,0]
        uav_positions = [[0,0]]
        for delta in uav_route:
            curr_position = [curr_position[0]+delta[0],curr_position[1]+delta[1]]
            uav_positions.append([curr_position[0],curr_position[1]])
        res.append(uav_positions)
    return res

def process_file(file_path,i):
    with open(file_path,'r') as file:
        lines = file.readlines()
        routes = list(map(lambda x: list(map(int,x.removesuffix('\n').split(' '))),lines))
        cartesian_route = delta_to_cartesian(move_to_delta(routes))
        string_route = json_dump(cartesian_route)
    with open(f"inputs/newCartesian/input/{i}.json",'w') as out_file:
        out_file.write(string_route)

def get_all_files(directory):
    # Recursively list all files in the directory and its subdirectories
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames]

directory = 'output/newCartesian/pather/generated_paths'
all_files = get_all_files(directory)

with tqdm(total=len(all_files), desc="Processing Files") as pbar:
    for i, file_path in enumerate(all_files):
        process_file(file_path, i)
        pbar.update(1)
print("Processing complete.")