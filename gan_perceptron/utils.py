from os import listdir,path,mkdir
from torch import float32, ones as torchOnes, tensor, zeros as torchZeros, randn,Tensor
from torch.utils.data import DataLoader, TensorDataset

from env_parser import Env

def label_real(size:int):
    env = Env.get_instance()
    return torchOnes(size,1).to(env.DEVICE)

def label_fake(size:int):
    env = Env.get_instance()
    return torchZeros(size,1).to(env.DEVICE)

def create_noise(size:int):
    env = Env.get_instance()
    return randn(size,env.NOISE_DIMENSION).to(env.DEVICE)

def load_dataset():
    env = Env.get_instance()
    if(not path.exists('./input')):
        raise AssertionError('Input directory should exist')
    subdirs = listdir('./input')
    all_file_routes:list[list[list[float]]] = []
    for i in subdirs:
        files = listdir(f"./input/{i}")
        for j in files:
            file = open(f"./input/{i}/{j}",'r')
            file_lines = file.readlines()
            file.close()
            file_routes = list(map(lambda x : list(map(float, x.split(' '))),file_lines))
            all_file_routes.append(file_routes)
    files_tensor_routes = (tensor(all_file_routes,dtype=float32) / 4 - 1).to(env.DEVICE)
    _labels = torchZeros(len(files_tensor_routes)).to(env.DEVICE)
    files_dataset = TensorDataset(files_tensor_routes,_labels)
    route_loader = DataLoader(files_dataset,batch_size=env.BATCH_SIZE,shuffle=True)
    return route_loader

def output_to_moves(route:Tensor) -> Tensor:
    return ((route + 1) * 4).round()

def tensor_to_file(tensor_routes:Tensor,file_name:str):
    route_samples:list[list[list[float]]] = tensor_routes.tolist()
    for (i,sample) in enumerate(route_samples):
        file = open(f"{file_name}.{i}.txt",'w')
        for route in sample:
            for move in route:
                file.write(f"{str(int(move))} ")
            file.write('\n')
        file.close()

def save_progress(g_losses,d_losses,path,epoch):
    
    pass