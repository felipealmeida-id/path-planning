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
    root = f'./inputs/{env.PY_ENV}/input'
    if not path.exists(root):
        raise AssertionError('Input directory should exist')
    subdirs = listdir(root)
    all_file_routes:list[list[list[float]]] = []
    for i in subdirs:
        files = listdir(f"{root}/{i}")
        for j in files:
            file = open(f"{root}/{i}/{j}",'r')
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

def save_progress(g_losses:list[float],d_losses:list[float],eval_avgs:list[float],epoch:int):
    from env_parser import Env
    env = Env.get_instance()
    with open(f"./output/{env.PY_ENV}/gan/summary.txt",'a') as file:
        for i,(g_loss,d_loss,eval_avg) in enumerate(zip(g_losses,d_losses,eval_avgs)):
            line = f"Epoch: {epoch+i} eval: {eval_avg} g_loss: {g_loss} d_loss: {d_loss}\n"
            file.write(line)
        file.close()
    g_losses.clear()
    d_losses.clear()
    eval_avgs.clear()