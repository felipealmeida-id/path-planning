from torch import FloatTensor, no_grad, randn
import torch
from env_parser import Env
from .generator import Generator
from .discriminator import Discriminator
from downscaler.downscaler import Downscaler
from downscaler.nn_down import NeuralDownscaler
from evaluator.main import evaluateGAN
from time import time
from .approaches import EvaluatorModuleApproach
from .utils import tensor_to_routes


def gan_perceptron():
    from .utils import load_dataset,checkpoint

    env = Env.get_instance()
    route_loader = load_dataset()
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_eval_avg = []
    discriminator = Discriminator().to(env.DEVICE)
    generator = Generator().to(env.DEVICE)
    downscaler_nn = NeuralDownscaler().to(env.DEVICE)
    # downscaler_nn.custom_train(randn((800,)).to(env.DEVICE),randn((400,)).to(env.DEVICE))
    downscaler_nn.load_pretrained_model()
    for epoch in range(env.EPOCHS):
        start = time()
        d_loss, g_loss, eval_avg = train_epoch(
            epoch, route_loader, discriminator, generator,downscaler_nn
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



def train_epoch(epoch:int, route_loader, Disc:Discriminator, Gen:Generator,DSNN:NeuralDownscaler):
    from .utils import create_noise

    def downscale(data_to_downscale):
        DSNN.eval()
        return DSNN(data_to_downscale)

    env = Env.get_instance()
    d_loss_acum = 0
    g_loss_acum = 0
    for _, (routes, _) in enumerate(route_loader):
        routes = routes.to(env.DEVICE)
        curr_batch_size = routes.size(0)
        for _ in range(env.K):
            data_real = routes
            # data_fake are 30x30 routes
            data_fake = Gen(create_noise(curr_batch_size))
            denormalized_data_fake = ((data_fake + 1 ) * (env.ENVIRONMENT_X_AXIS/2))
            downscaled_data_fake = downscale(denormalized_data_fake)
            downscaled_data_fake = downscaled_data_fake / (env.ENVIRONMENT_X_AXIS/2) - 1
            # downscaled_data_fake = data_fake
            d_loss = Disc.custom_train(data_real, downscaled_data_fake)
            # d_loss = Disc.custom_train(data_real, data_fake)
            d_loss_acum += d_loss
        # data_fake are 30x30 routes
        data_fake = Gen(create_noise(curr_batch_size))
        # move_list = output_to_moves(data_fake).tolist()
        # evaluatorModules = EvaluatorModuleApproach.get_instance().get_evaluator_modules(
        #     epoch
        # )
        # evaluations = list(map(lambda x: evaluateGAN(x, evaluatorModules), move_list))
        # all will evaluate 0
        denormalized_data_fake = ((data_fake + 1 ) * (env.ENVIRONMENT_X_AXIS/2))
        downscaled_data_fake = downscale(denormalized_data_fake)
        downscaled_data_fake = downscaled_data_fake / (env.ENVIRONMENT_X_AXIS/2) - 1
        # evaluations = [0] * curr_batch_size
        jump_penalty_tensor = jump_penalty(downscaled_data_fake)
        evaluations = jump_penalty_tensor.tolist()
        eval_tensor = FloatTensor(evaluations).to(env.DEVICE)
        # downscaled_data_fake = data_fake
        g_loss = Gen.custom_train(Disc, downscaled_data_fake, eval_tensor, epoch)
        # g_loss = Gen.custom_train(Disc, data_fake, eval_tensor, epoch)
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
    env = Env.get_instance()
    # Copy tensor to avoid changing the original
    tensor = tensor.clone()
    # Calculate the differences along the sequence_length axis
    tensor = tensor_to_routes(tensor)
    # print(tensor)
    diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    
    # Define the legal moves
    legal_horizontal_vertical = ((torch.abs(diff[..., 0]) <= 1) & (diff[..., 1] == 0)) | ((diff[..., 0] == 0) & (torch.abs(diff[..., 1]) <= 1))
    legal_diagonal = (torch.abs(diff[..., 0]) == 1) & (torch.abs(diff[..., 1]) == 1)
    
    # Identify illegal moves
    illegal_moves = ~(legal_horizontal_vertical | legal_diagonal)
    
    # Count the number of illegal moves for each batch
    count_illegal = torch.sum(illegal_moves, dim=(1,2))

    # Now 0 is the worst and 1 is the best
    count_illegal = 1 - count_illegal / (env.HR_TOTAL_TIME - 1)
    
    return count_illegal