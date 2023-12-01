from torch import empty, norm, sum
from torch.nn import Module, Linear, Sequential, Tanh, LeakyReLU, Sigmoid, LSTM
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
        self.optimizer = Adam(self.parameters(), lr=env.G_LEARN_RATE)

    def _build_model(self):
        env = Env.get_instance()
        model = Sequential(
            Linear(self.noise_dim, 256),
            LeakyReLU(0.2),
            LSTM(input_size=256, hidden_size=512, batch_first=True),  # First LSTM layer
            LeakyReLU(0.2),
            Linear(512, 1024),
            LeakyReLU(0.2),
            Linear(1024, env.UAV_AMOUNT * (env.TOTAL_TIME * 2)),
            Tanh(),
        )
        self._initialize_weights(model)
        return model

    def forward(self, x):
        env = Env.get_instance()
        x = self.main[0](x)  # Linear
        x = self.main[1](x)  # LeakyReLU
        x, _ = self.main[2](x)  # First LSTM, ignoring hidden states
        for i in range(3, len(self.main)):
            x = self.main[i](x)
        out = x.view(-1, env.UAV_AMOUNT, env.TOTAL_TIME, 2)
        return out

    # He uniform initialization
    def _initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    constant_(m.bias.data, 0)
        return model

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
        # print(eval_tensor.grad)
        self.optimizer.step()
        return loss


