from torch import FloatTensor, save
from env_parser import Env
from .generator import Generator
from .discriminator import Discriminator
from downscaler.downscaler import Downscaler
from evaluator.main import evaluateGAN
from time import time
from .approaches import EvaluatorModuleApproach


def gan_perceptron():
    from .utils import load_dataset,checkpoint

    env = Env.get_instance()
    route_loader = load_dataset()
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_eval_avg = []
    discriminator = Discriminator().to(env.DEVICE)
    generator = Generator().to(env.DEVICE)
    downscaler = Downscaler()
    for epoch in range(env.EPOCHS):
        start = time()
        d_loss, g_loss, eval_avg = train_epoch(
            epoch, route_loader, discriminator, generator, downscaler
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



def train_epoch(epoch:int, route_loader, discriminator:Discriminator, generator:Generator, downscaler: Downscaler):
    from .utils import create_noise, output_to_moves

    env = Env.get_instance()
    d_loss_acum = 0
    g_loss_acum = 0
    for _, (images, _) in enumerate(route_loader):
        images = images.to(env.DEVICE)
        curr_batch_size = images.size(0)
        for _ in range(env.K):
            # data_fake are 30x30 images
            data_fake = generator(create_noise(curr_batch_size))
            data_real = images
            downscaled_data_fake = (
                FloatTensor(
                    list(
                        map(
                            downscaler.downscale_trajectory,
                            (output_to_moves(data_fake).tolist()),
                        )
                    )
                )
                / 4
                - 1
            ).to(env.DEVICE)
            d_loss = discriminator.custom_train(data_real, downscaled_data_fake)
            d_loss_acum += d_loss
        # data_fake are 30x30 images
        data_fake = generator(create_noise(curr_batch_size))
        move_list = output_to_moves(data_fake).tolist()
        evaluatorModules = EvaluatorModuleApproach.get_instance().get_evaluator_modules(
            epoch
        )
        evaluations = list(map(lambda x: evaluateGAN(x, evaluatorModules), move_list))
        downscaled_data_fake = (
            FloatTensor(list(map(downscaler.downscale_trajectory, move_list))) / 4 - 1
        ).to(env.DEVICE)
        eval_tensor = FloatTensor(evaluations).to(env.DEVICE)
        g_loss = generator.custom_train(
            discriminator, downscaled_data_fake, eval_tensor, epoch
        )
        g_loss_acum += g_loss
        eval_avg = eval_tensor.mean()
        eval_tensor.detach()
        del eval_tensor
    return (
        float(d_loss_acum) / len(route_loader),
        float(g_loss_acum) / len(route_loader),
        eval_avg,
    )
