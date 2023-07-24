from torch.nn import Module, Linear, Sequential, Tanh, LeakyReLU
from torch.optim import Optimizer

from env_parser import EPOCHS,TOTAL_TIME,UAV_AMOUNT
from discriminator import Discriminator
from custom_loss import CustomLoss
from utils import label_real


class Generator(Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.main = Sequential(
            Linear(self.noise_dim, 256),
            LeakyReLU(0.2),
            Linear(256, 512),
            LeakyReLU(0.2),
            Linear(512, 1024),
            LeakyReLU(0.2),
            Linear(1024, UAV_AMOUNT * TOTAL_TIME),
            Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(-1, UAV_AMOUNT, TOTAL_TIME)


def train_generator(discriminator: Discriminator, g_optimizer: Optimizer, data_fake, eval_tensor, epoch):
    curr_batch_size = data_fake.size(0)
    real_label = label_real(curr_batch_size)
    g_optimizer.zero_grad()
    output = discriminator(data_fake)
    evalWeight = epoch/EPOCHS
    regularWeight = (EPOCHS-epoch)/EPOCHS
    loss_fun = CustomLoss(eval_tensor, evalWeight, regularWeight)
    loss = loss_fun(output, real_label)
    loss.backward()
    g_optimizer.step()
    return loss
