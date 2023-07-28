from torch import FloatTensor,Module,save
# from time import time
from env_parser import Env
from .generator import Generator
from .discriminator import Discriminator

def gan_perceptron():
    from .utils import load_dataset,save_progress
    env = Env.get_instance()
    route_loader = load_dataset()
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_eval_avg = []
    discriminator = Discriminator().to(env.DEVICE)
    generator = Generator(env.NOISE_DIMENSION).to(env.DEVICE)
    for epoch in range(env.EPOCHS):
        d_loss,g_loss,eval_avg = train_epoch(epoch,route_loader,discriminator,generator)
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
        epoch_eval_avg.append(eval_avg)
        if epoch % 32 == 31:
            save(discriminator.state_dict(),)
            save(generator.state_dict(),)
            save_progress(epoch_g_losses,epoch_d_losses,epoch_eval_avg,epoch)

def train_epoch(epoch,route_loader,discriminator:Module,generator:Module):
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
        # TODO: Add missing evaluator
        evaluations = list(map(lambda _:1, move_list))
        eval_tensor = FloatTensor(evaluations).to(env.DEVICE)
        g_loss = generator.custom_train(discriminator,data_fake,eval_tensor,epoch)
        eval_avg = eval_tensor.mean()
        eval_tensor.detach()
        del eval_tensor
    return float(d_loss)/len(route_loader),float(g_loss)/len(route_loader),eval_avg