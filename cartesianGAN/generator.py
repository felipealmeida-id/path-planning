from torch import empty
from torch.nn import Module, Linear, Sequential, Tanh, LeakyReLU, Sigmoid
from torch.nn.init import constant_, kaiming_normal_
from torch.optim import Optimizer, Adam, SGD

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
        self.main = self._build_model()
        # Using SGD with momentum instead of Adam
        self.optimizer = SGD(self.parameters(), lr=env.G_LEARN_RATE, momentum=0.9)

    def _build_model(self):
        env = Env.get_instance()
        model = Sequential(
            Linear(self.noise_dim, 256),
            LeakyReLU(0.2),
            Linear(256, 512),
            LeakyReLU(0.2),
            Linear(512, 1024),
            LeakyReLU(0.2),
            Linear(1024, env.UAV_AMOUNT * (env.TOTAL_TIME * 2)),
            Sigmoid(),
        )
        self._initialize_weights(model)
        return model

    def _initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, Linear):
                kaiming_normal_(m.weight.data)  # Inicialización de He
                if m.bias is not None:
                    constant_(m.bias.data, 0)
        return model

    def forward(self, x):
        env = Env.get_instance()
        # Multiply by 30 to get the correct range
        out =  self.main(x).view(-1, env.UAV_AMOUNT, env.TOTAL_TIME, 2) *30
        out = out.round()
        return out
        
    def custom_train(self, discriminator, fake_data, eval_tensor, epoch):
        self.optimizer.zero_grad()

        # WGAN loss
        loss = -discriminator(fake_data).mean()

        # Here you can add any other loss components, like the evaluation loss.
        # For this example, I'm ignoring eval_tensor, but you can integrate it as needed.

        loss.backward()
        self.optimizer.step()

        return loss.item()
    


    # def custom_train(
    #     self, discriminator: Discriminator, data_fake, eval_tensor, epoch: int
    # ):
    #     curr_batch_size = data_fake.size(0)
    #     real_label = label_real(curr_batch_size)
    #     self.optimizer.zero_grad()
    #     output = discriminator(data_fake)
    #     eval_weight, regular_weight = WeightApproach.get_instance().get_weights(epoch)
    #     self.loss_fun.adjust_weights(eval_weight, regular_weight)
    #     self.loss_fun.set_evaluations(eval_tensor)
    #     loss = self.loss_fun(output, real_label)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss
