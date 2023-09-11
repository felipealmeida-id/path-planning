import sys
from json import load


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
decode_map = {tuple(v): k for k, v in move_encode_map.items()}

def decode(encoded_values):
    decoded_values = []
    for item in encoded_values:
        decoded_values.append([decode_map[tuple(subitem)] for subitem in item])
    return decoded_values

file_path = sys.argv[1]
file = open(file_path,'r')
encoded_data = load(file)
decoded_data = decode(encoded_data)
file_path = "./output.txt"
with open(file_path, 'w') as file:
    for array in decoded_data:
        file.write(' '.join(map(str, array)) + '\n')