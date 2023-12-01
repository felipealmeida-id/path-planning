from os import listdir,path,mkdir
from torch import float32, ones as torchOnes, tensor, zeros as torchZeros, randn,Tensor,save
from torch.utils.data import DataLoader, TensorDataset
import json

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
    all_file_routes:list[list[list[list[float]]]] = []
    files = listdir(f"{root}")
    for j in files:
        with open(f"{root}/{j}") as j:
            file_lines = json.load(j)
            all_file_routes.append(file_lines)
    files_tensor_routes = (tensor(all_file_routes,dtype=float32) / (env.ENVIRONMENT_X_AXIS/2) - 1).to(env.DEVICE)
    _labels = torchOnes(len(files_tensor_routes)).to(env.DEVICE)
    files_dataset = TensorDataset(files_tensor_routes,_labels)
    route_loader = DataLoader(files_dataset,env.BATCH_SIZE,shuffle=True)
    return route_loader

def output_to_moves(route:Tensor) -> Tensor:
    return ((route + 1) * 4).round()

def tensor_to_file(tensor_routes:Tensor,file_name:str):
    # route_samples:list[list[list[float]]] = tensor_routes.tolist()
    # for (i,sample) in enumerate(route_samples):
    #     file = open(f"{file_name}.{i}.txt",'w')
    #     for route in sample:
    #         for move in route:
    #             file.write(f"{str(int(move))} ")
    #         file.write('\n')
    #     file.close()
    route_samples:list[list[list[list[float]]]] = tensor_routes.tolist()
    for (i,sample) in enumerate(route_samples):
        json.dump(sample,open(f"{file_name}.{i}.txt",'w'))

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

def checkpoint(discriminator,generator,epoch_g_losses, epoch_d_losses, epoch_eval_avg, epoch):
    from env_parser import Env
    env = Env.get_instance()
    save(discriminator.state_dict(),f"./output/{env.PY_ENV}/gan/discriminator/d_{epoch}",)
    save(generator.state_dict(), f"./output/{env.PY_ENV}/gan/generator/g_{epoch}")
    save_progress(epoch_g_losses, epoch_d_losses, epoch_eval_avg, epoch)
    noise = create_noise(3)
    generated_img = generator(noise).to(env.DEVICE).detach()
    scaled_tensor = (generated_img + 1 ) * (env.ENVIRONMENT_X_AXIS/2)
    scaled_tensor = scaled_tensor.round().long()
    tensor_to_file(scaled_tensor, f"output/{env.PY_ENV}/gan/generated_imgs/test.{epoch}")