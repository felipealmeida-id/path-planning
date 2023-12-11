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

            # denormalized_data_fake = ((data_fake + 1 ) * (env.HR_ENVIRONMENT_X_AXIS/2))
            downscaled_data_fake = downscale(data_fake)
            # downscaled_data_fake = downscaled_data_fake / (env.ENVIRONMENT_X_AXIS/2) - 1

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

        # denormalized_data_fake = ((data_fake + 1 ) * (env.HR_ENVIRONMENT_X_AXIS/2))
        downscaled_data_fake = downscale(data_fake)
        # downscaled_data_fake = downscaled_data_fake / (env.ENVIRONMENT_X_AXIS/2) - 1

        # evaluations = [0] * curr_batch_size
        # downscaled_data_fake = data_fake
        jump_penalty_tensor = evaluar_trayectorias(downscaled_data_fake)
        evaluations = jump_penalty_tensor.tolist()
        eval_tensor = FloatTensor(evaluations).to(env.DEVICE)
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


def evaluar_trayectorias(trayectorias_batch):
    puntuacion_maxima_por_trayectoria = 100
    penalizacion_por_salto = 5  # Penalización ajustable

    # Inicializar un tensor para las puntuaciones de cada lote
    puntuaciones = torch.zeros(trayectorias_batch.shape[0], dtype=torch.float)

    for i in range(trayectorias_batch.shape[0]):  # Itera sobre cada lote
        puntuacion_lote = 0
        for j in range(trayectorias_batch.shape[1]):  # Itera sobre cada UAV
            trayectoria = trayectorias_batch[i, j].float()

            # Calcula diferencias entre puntos consecutivos
            diferencias = torch.abs(torch.diff(trayectoria, dim=0))

            # Verifica si hay algún salto donde la diferencia en alguna coordenada sea mayor a 1
            saltos = diferencias > 1
            penalizaciones = saltos.sum() * penalizacion_por_salto

            # Calcula la puntuación para esta trayectoria
            puntuacion = puntuacion_maxima_por_trayectoria - penalizaciones
            puntuacion = torch.clamp(puntuacion, min=0)

            puntuacion_lote += puntuacion

        # Normaliza la puntuación del lote
        puntuacion_normalizada = (puntuacion_lote / (trayectorias_batch.shape[1] * puntuacion_maxima_por_trayectoria)) * 100
        puntuaciones[i] = puntuacion_normalizada

    return puntuaciones/100


def jump_penalty(tensorParam):    
    env = Env.get_instance()
    # Copy tensor to avoid changing the original
    tensor = tensorParam.clone().detach()
    # Calculate the differences along the sequence_length axis
    tensor = tensor_to_routes(tensor)
    print(tensor.shape)
    # print(tensor)
    diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    
    # Define the legal moves
    legal_horizontal_vertical = ((torch.abs(diff[..., 0]) <= 1) & (diff[..., 1] == 0)) | ((diff[..., 0] == 0) & (torch.abs(diff[..., 1]) <= 1))
    legal_diagonal = (torch.abs(diff[..., 0]) == 1) & (torch.abs(diff[..., 1]) == 1)
    
    # Identify illegal moves
    illegal_moves = ~(legal_horizontal_vertical | legal_diagonal)
    
    # Count the number of illegal moves for each batch
    count_illegal = torch.sum(illegal_moves, dim=(1,2))

    count_illegal = count_illegal / (env.TOTAL_TIME - 1)
    print(count_illegal)

    # Now 0 is the worst and 1 is the best
    count_illegal = 1 - count_illegal

    
    return count_illegal