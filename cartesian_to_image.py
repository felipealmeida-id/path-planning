import os
import json
from tqdm import tqdm

def process_file(file_path,i):
    matrix = [[0 for _ in range(15)] for _ in range(15)]
    with open(file_path, 'r') as file:
        data = json.load(file)
        for x,y in data[0]:
            matrix[x][y] = 1
    with open(f"temp_out/{i}.json",'w') as out_file:
        json.dump(matrix,out_file)

def get_all_files(directory):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames]

directory = 'to_transform'
all_files = get_all_files(directory)

with tqdm(total=len(all_files), desc="Processing Files") as pbar:
    for i, file_path in enumerate(all_files):
        process_file(file_path, i)
        pbar.update(1)

print("Processing complete.")