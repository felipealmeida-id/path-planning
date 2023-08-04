from torch import Tensor, randn
from torch.nn import Module, Linear, Sequential, LeakyReLU, Dropout, Sigmoid, BCELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer,Adam

from env_parser import Env
from .utils import label_fake, label_real

class Discriminator(Module):
    loss_function:_Loss = BCELoss()
    optimizer:Optimizer

    def __init__(self):
        env = Env.get_instance()
        super(Discriminator, self).__init__()
        self.n_input = env.UAV_AMOUNT * env.TOTAL_TIME
        self.main = Sequential(
          Linear(self.n_input, 1024),
          LeakyReLU(0.2),
          Dropout(0.4),
          Linear(1024, 512),
          LeakyReLU(0.2),
          Dropout(0.4),
          Linear(512, 256),
          LeakyReLU(0.2),
          Dropout(0.4),
          Linear(256, 1),
          Sigmoid(),
        )
        self.optimizer = Adam(self.parameters(),lr=env.D_LEARN_RATE)

    def forward(self, x):
        env = Env.get_instance()
        x= x + randn(x.size()).to(env.DEVICE) * 0.1
        x = x.view(-1, self.n_input)
        return self.main(x)

    def custom_train(self,data_real:Tensor, data_fake:Tensor):
        curr_batch_size = data_real.size(0)
        real_label = label_real(curr_batch_size)
        fake_label = label_fake(curr_batch_size)
        self.optimizer.zero_grad()
        output_real = self(data_real)
        real_smooth_label = self.label_smoothing(real_label)
        loss_real = self.loss_function(output_real, real_smooth_label)
        output_fake = self(data_fake)
        fake_smooth_label = self.label_smoothing(fake_label)
        loss_fake = self.loss_function(output_fake, fake_smooth_label)
        loss_real.backward()
        loss_fake.backward()
        self.optimizer.step()
        return loss_real + loss_fake

    def label_smoothing(self,target, smoothing=0.2):
        return target * (1 - smoothing) + 0.5 * smoothing
