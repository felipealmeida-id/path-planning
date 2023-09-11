from json import dump as json_dump
from os import listdir
from os.path import isfile

# [R,D,L,U]
move_encode_map = {
    0:[0,0,0,0],
    1:[1,0,0,0],
    2:[1,1,0,0],
    3:[0,1,0,0],
    4:[0,1,1,0],
    5:[0,0,1,0],
    6:[0,0,1,1],
    7:[0,0,0,1],
    8:[1,0,0,1]
}

move_decode_map = {tuple(encoded): decoded for decoded, encoded in move_encode_map.items()}


def encode_move(move:int):
    return move_encode_map[move]

def encode_list(moves:list[int]) -> list[list[int]]:
    return list(map(encode_move,moves))

def line_to_route(line:str):
    return list(map(int,line.split(' ')))

def encode_file(file_name:str,out_file_name:str):
    file = open(file_name,'r')
    line_routes = file.readlines()
    file.close()
    out_file = open(out_file_name,'w')
    encoded_lines = [encode_list(line_to_route(line_route)) for line_route in line_routes]
    json_dump(encoded_lines,out_file)

def iterate_over_dir(directory:str):
    file_names = listdir(directory)
    file_names = [f"{directory}/{file}" for file in file_names if isfile(f"{directory}/{file}")]
    for file_name in file_names:
        strip = len(directory)
        out_file = f"./encode/output/{file_name[strip:]}"
        encode_file(file_name,out_file)