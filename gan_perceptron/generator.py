from torch import empty
from torch.nn import Module, Linear, Sequential, Tanh, LeakyReLU
from torch.nn.init import constant_
from torch.optim import Optimizer, Adam

from env_parser import Env
from .discriminator import Discriminator
from .custom_loss import CustomLoss
from .utils import label_real

from .approaches import WeightApproach


class Generator(Module):
    optimizer: Optimizer
    loss_fun: CustomLoss = CustomLoss(empty(1), 0, 0)

    def __init__(self):
        env = Env.get_instance()
        super(Generator, self).__init__()
        self.noise_dim = env.NOISE_DIMENSION
        self.main = Sequential(
            Linear(self.noise_dim, 256),
            LeakyReLU(0.2),
            Linear(256, 512),
            LeakyReLU(0.2),
            Linear(512, 1024),
            LeakyReLU(0.2),
            Linear(1024, env.UAV_AMOUNT * (env.TOTAL_TIME * 2)),
            Tanh(),
        )
        self.optimizer = Adam(self.parameters(), lr=env.G_LEARN_RATE)

    def forward(self, x):
        env = Env.get_instance()
        return self.main(x).view(-1, env.UAV_AMOUNT, env.TOTAL_TIME * 2)

    def custom_train(
        self, discriminator: Discriminator, data_fake, eval_tensor, epoch: int
    ):
        env = Env.get_instance()
        curr_batch_size = data_fake.size(0)
        real_label = label_real(curr_batch_size)
        self.optimizer.zero_grad()
        output = discriminator(data_fake)
        eval_weight, regular_weight = WeightApproach.get_instance().get_weights(epoch)
        self.loss_fun.adjust_weights(eval_weight, regular_weight)
        self.loss_fun.set_evaluations(eval_tensor)
        loss = self.loss_fun(output, real_label)
        loss.backward()
        self.optimizer.step()
        return loss
