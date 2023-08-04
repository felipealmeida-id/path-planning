from torch import FloatTensor,save
from env_parser import Env
from .generator import Generator
from .discriminator import Discriminator
from evaluator.main import evaluateGAN
import cProfile
import pstats
import csv

def gan_perceptron():
    from .utils import load_dataset,save_progress
    env = Env.get_instance()
    route_loader = load_dataset()
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_eval_avg = []
    discriminator = Discriminator().to(env.DEVICE)
    generator = Generator().to(env.DEVICE)
    for epoch in range(env.EPOCHS):
        d_loss,g_loss,eval_avg = train_epoch(epoch,route_loader,discriminator,generator)
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
        epoch_eval_avg.append(eval_avg)
        if epoch % 32 == 31:
            save(discriminator.state_dict(),f"./output/{env.PY_ENV}/discriminator/d_{epoch}")
            save(generator.state_dict(),f"./output/{env.PY_ENV}/generator/g_{epoch}")
            save_progress(epoch_g_losses,epoch_d_losses,epoch_eval_avg,epoch)

def train_epoch(epoch,route_loader,discriminator,generator):
    from .utils import create_noise,output_to_moves
    env = Env.get_instance()
    for _, (images, _) in enumerate(route_loader):
        images = images.to(env.DEVICE)
        curr_batch_size = images.size(0)
        for _ in range(env.K): 
            data_fake = generator(create_noise(curr_batch_size))
            data_real = images
            d_loss = discriminator.custom_train(data_real,data_fake)
        data_fake = generator(create_noise(curr_batch_size))
        move_list = output_to_moves(data_fake).tolist()
        evaluations = list(map(evaluateGAN, move_list))
        eval_tensor = FloatTensor(evaluations).to(env.DEVICE)
        g_loss = generator.custom_train(discriminator,data_fake,eval_tensor,epoch)
        eval_avg = eval_tensor.mean()
        eval_tensor.detach()
        del eval_tensor
    return float(d_loss)/len(route_loader),float(g_loss)/len(route_loader),eval_avg

def profiler():
    pr = cProfile.Profile()
    pr.enable()
    gan_perceptron()
    pr.disable()
    ps = pstats.Stats(pr)
    with open('profile_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['function_name', 'ncalls', 'tottime', 'cumtime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for func, info in ps.stats.items():
            writer.writerow({'function_name': func, 'ncalls': info[0], 'tottime': info[2], 'cumtime': info[3]})
