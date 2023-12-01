from torch import FloatTensor, no_grad, randn, load, norm, sum, abs
from env_parser import Env
from .generator import Generator
from .discriminator import Discriminator
from downscaler.downscaler import Downscaler
from evaluator.main import evaluateGAN
from time import time
from .approaches import EvaluatorModuleApproach
import os
from .utils import tensor_to_routes

def gan_cartesian():
    from .utils import load_dataset,checkpoint

    env = Env.get_instance()
    discriminator = Discriminator().to(env.DEVICE)
    # If there is a discriminator file at the same level of this file, load it
    if os.path.isfile('discriminator'):
        discriminator.load_state_dict(load('discriminator'))
        print("Discriminator loaded")
    generator = Generator().to(env.DEVICE)
    # If there is a generator file, load it
    if os.path.isfile('generator'):
        generator.load_state_dict(load('generator'))
        print("Generator loaded")
    print("GAN training")
    route_loader = load_dataset()
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_eval_avg = []
        

    # downscaler = Downscaler()
    # downscaler_nn = DownscalerNN().to(env.DEVICE)
    # downscaler_nn.custom_train(randn((800,)).to(env.DEVICE),randn((400,)).to(env.DEVICE))
    # downscaler_nn.load_pretrained_model()
    for epoch in range(env.EPOCHS):
        start = time()
        # d_loss, g_loss, eval_avg = train_epoch(
        #     epoch, route_loader, discriminator, generator, downscaler,downscaler_nn
        # )
        d_loss, g_loss, eval_avg = train_epoch(
            epoch, route_loader, discriminator, generator
        )
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
        epoch_eval_avg.append(eval_avg)
        end = time()
        print(
            f"Epoch {epoch} | D loss: {d_loss} | G loss: {g_loss} | Eval avg: {eval_avg} | Time: {end - start}"
        )
        if epoch % 10 == 0:
            checkpoint(discriminator,generator,epoch_g_losses, epoch_d_losses, epoch_eval_avg, epoch)



def train_epoch(epoch:int, route_loader, Disc:Discriminator, Gen:Generator,
                #  DS: Downscaler,DSNN:DownscalerNN
                 ):
    from .utils import create_noise

    # def downscale(data_to_downscale):
    #     with no_grad():
    #         return DSNN(data_to_downscale)

    env = Env.get_instance()
    d_loss_acum = 0
    g_loss_acum = 0
    for [images] in route_loader:
        images = images.to(env.DEVICE)
        curr_batch_size = images.size(0)
        for _ in range(env.K):
            # data_fake are 30x30 images
            data_fake = Gen(create_noise(curr_batch_size))
            data_real = images
            # downscaled_data_fake = downscale(data_fake)
            d_loss = Disc.custom_train(data_real, data_fake)
            d_loss_acum += d_loss
        # data_fake are 30x30 images
        data_fake = Gen(create_noise(curr_batch_size))
        # move_list = data_fake.tolist()
        # evaluatorModules = EvaluatorModuleApproach.get_instance().get_evaluator_modules(
        #     epoch
        # )
        # evaluations = list(map(lambda x: evaluateGAN(x, evaluatorModules), move_list))
        # # all will evaluate 0

        # # downscaled_data_fake = downscale(data_fake)
        invalid_jumps = jump_penalty(data_fake)
        normalized_penalty = invalid_jumps / (env.UAV_AMOUNT * env.TOTAL_TIME)
        eval_tensor = 1 - normalized_penalty

        # evaluations = [0] * curr_batch_size
        # eval_tensor = FloatTensor(evaluations).to(env.DEVICE)

        g_loss = Gen.custom_train(Disc, data_fake, eval_tensor, epoch)
        g_loss_acum += g_loss
        eval_avg = eval_tensor.mean()
        eval_tensor.detach()
        del eval_tensor
    return (
        float(d_loss_acum) / len(route_loader) / env.K,
        float(g_loss_acum) / len(route_loader),
        eval_avg,
    )


def jump_penalty(tensor):    
    # Calculate the differences along the sequence_length axis
    tensor = tensor_to_routes(tensor)
    # print(tensor)
    diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    
    # Define the legal moves
    legal_horizontal_vertical = ((abs(diff[..., 0]) <= 1) & (diff[..., 1] == 0)) | ((diff[..., 0] == 0) & (abs(diff[..., 1]) <= 1))
    legal_diagonal = (abs(diff[..., 0]) == 1) & (abs(diff[..., 1]) == 1)
    
    # Identify illegal moves
    illegal_moves = ~(legal_horizontal_vertical | legal_diagonal)
    
    # Count the number of illegal moves for each batch
    count_illegal = sum(illegal_moves, dim=(1,2))
    
    return count_illegal

# def jump_penalty(batch):
#     # Assuming batch is a PyTorch tensor with shape [batch_size, num_drones, sequence_length, 2]
    
#     # Split the batch into coordinates at time t and t+1
#     coords_t = batch[:, :, :-1, :]
#     coords_t1 = batch[:, :, 1:, :]
    
#     # Compute differences in coordinates
#     diff = coords_t1 - coords_t
    
#     # Calculate the distances
#     distances = norm(diff, dim=-1)
    
#     # Identify where distance is greater than 1
#     invalid_jumps = distances > 1

#     # # Now turn the batch_size, num_drones, sequence_length tensor into a batch_size tensor
#     invalid_jumps = sum(invalid_jumps.int(), dim=(1,2))

    
#     total_penalty = distances
    
#     return invalid_jumps
