from os import listdir,path,mkdir
from torch import float32, ones as torchOnes, tensor, zeros as torchZeros, randn,Tensor,save
from torch.utils.data import DataLoader, TensorDataset
from json import load, dump

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
    all_file_routes: list[list[list[float]]] = []
    
    for i in subdirs:
        file_path = path.join(root, i)
        if path.isfile(file_path):
            with open(file_path, 'r') as file_obj:
                file_data = load(file_obj)
                all_file_routes.append(file_data)
    
    # we are going to try without normalization
    files_tensor_routes = tensor(all_file_routes, dtype=float32).to(env.DEVICE)
    files_dataset = TensorDataset(files_tensor_routes)
    route_loader = DataLoader(files_dataset, env.BATCH_SIZE, shuffle=True)
    
    return route_loader

def tensor_to_file(tensor_routes:Tensor,file_name:str):
    # convert all floats to rounded integers
    tensor_routes = tensor_routes.round().int()
    # Array con x samples de array de uavs que tienen un array de posiciones (x,y) que son floats
    route_samples:list[list[list[list[float]]]] = tensor_routes.tolist()

    for (i,sample) in enumerate(route_samples):
        with open(f"{file_name}.{i}.txt",'w') as file:
            dump(sample,file)

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
    generated_imges = generator(noise).to(env.DEVICE).detach()
    tensor_to_file(generated_imges, f"output/{env.PY_ENV}/gan/generated_imgs/test.{epoch}")