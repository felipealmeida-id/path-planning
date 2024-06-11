from torch import FloatTensor,save
from env_parser import Env
from .generator import Generator
from .discriminator import Discriminator
from evaluator.main import evaluateGAN
from time import time
import cProfile
import pstats
import csv

def gan_perceptron():
    from .utils import load_dataset,checkpoint
    env = Env.get_instance()
    route_loader = load_dataset()
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_eval_avg = []
    discriminator = Discriminator().to(env.DEVICE)
    generator = Generator().to(env.DEVICE)
    for epoch in range(env.EPOCHS):
        start = time()
        d_loss,g_loss,eval_avg = train_epoch(epoch,route_loader,discriminator,generator)
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
        epoch_eval_avg.append(eval_avg)
        end = time()
        print(
            f"Epoch {epoch} | D loss: {d_loss} | G loss: {g_loss} | Eval avg: {eval_avg} | Time: {end - start}"
        )
        if epoch % 10 == 0:
            checkpoint(discriminator,generator,epoch_g_losses, epoch_d_losses, epoch_eval_avg, epoch)

def train_epoch(epoch,route_loader,discriminator,generator):
    from .utils import create_noise,output_to_moves
    env = Env.get_instance()
    d_loss_acum = 0
    g_loss_acum = 0
    eval_avg_acum = 0
    for _, (images, _) in enumerate(route_loader):
        images = images.to(env.DEVICE)
        curr_batch_size = images.size(0)
        for _ in range(env.K): 
            data_fake = generator(create_noise(curr_batch_size))
            data_real = images
            d_loss = discriminator.custom_train(data_real,data_fake)
            d_loss_acum += d_loss
        data_fake = generator(create_noise(curr_batch_size))
        move_list = output_to_moves(data_fake).tolist()
        evaluations = list(map(evaluateGAN, move_list))
        eval_tensor = FloatTensor(evaluations).to(env.DEVICE)
        g_loss = generator.custom_train(discriminator,data_fake,eval_tensor,epoch)

        g_loss_acum += g_loss
        eval_avg = eval_tensor.mean()
        eval_avg_acum += eval_avg
        eval_tensor.detach()
        del eval_tensor
    return float(d_loss_acum)/len(route_loader)/env.K,float(g_loss_acum)/len(route_loader),eval_avg_acum / len(route_loader)


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
