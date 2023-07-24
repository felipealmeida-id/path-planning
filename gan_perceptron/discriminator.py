from torch import Tensor, randn
from torch.nn import Module, Linear, Sequential, LeakyReLU, Dropout, Sigmoid, BCELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from env_parser import DEVICE,TOTAL_TIME,UAV_AMOUNT
from .utils import label_fake, label_real

class Discriminator(Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.n_input = UAV_AMOUNT * TOTAL_TIME
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

  def forward(self, x):
    x= x + randn(x.size()).to(DEVICE) * 0.1
    x = x.view(-1, self.n_input)
    return self.main(x)
  
def binary_cross_entropy_with_label_smoothing(input, target, smoothing=0.2):
  target = target * (1 - smoothing) + 0.5 * smoothing
  loss = BCELoss()(input, target)
  return loss


def train_discriminator(discriminator:Discriminator,loss_fun:_Loss, d_optimizer:Optimizer, 
                        data_real:Tensor, data_fake:Tensor):
  curr_batch_size = data_real.size(0)
  real_label = label_real(curr_batch_size)
  fake_label = label_fake(curr_batch_size)
  d_optimizer.zero_grad()
  output_real = discriminator(data_real)
  loss_real = binary_cross_entropy_with_label_smoothing(output_real, real_label)
  output_fake = discriminator(data_fake)
  loss_fake = binary_cross_entropy_with_label_smoothing(output_fake, fake_label)
  loss_real.backward()
  loss_fake.backward()
  d_optimizer.step()
  return loss_real + loss_fake
