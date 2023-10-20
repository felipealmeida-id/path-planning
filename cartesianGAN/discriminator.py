from torch import Tensor, randn
from torch.nn import Module, Linear, Sequential, LeakyReLU, Dropout, Sigmoid, BCELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam, SGD
from torch.nn import init

from env_parser import Env
from .utils import label_fake, label_real


class Discriminator(Module):
    loss_function: _Loss = BCELoss()
    optimizer: Optimizer

    def __init__(self):
        env = Env.get_instance()
        super(Discriminator, self).__init__()
        self.n_input = env.UAV_AMOUNT * env.TOTAL_TIME * 2
        self.main = self._build_model()
        self.optimizer = SGD(self.parameters(), lr=env.D_LEARN_RATE, momentum=0.9)

    def _build_model(self):
        model = Sequential(
            Linear(self.n_input, 1024),
            LeakyReLU(0.2),
            Dropout(0.2),
            Linear(1024, 512),
            LeakyReLU(0.2),
            Dropout(0.2),
            Linear(512, 256),
            LeakyReLU(0.2),
            Dropout(0.2),
            Linear(256, 1),
        )
        self._initialize_weights(model)
        return model

    def _initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, Linear):
                init.kaiming_normal_(m.weight.data)  # Inicialización de He
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
        return model

    def forward(self, x):
        env = Env.get_instance()
        x = x + randn(x.size()).to(env.DEVICE) * 0.1
        x = x.view(-1, self.n_input)
        return self.main(x)

    # Update to Discriminator's custom_train
    def custom_train(self, real_data, fake_data):
        self.optimizer.zero_grad()

        # Calculate the loss for real and fake data
        real_loss = self(real_data).mean()
        fake_loss = self(fake_data).mean()

        # WGAN loss
        loss = fake_loss - real_loss

        loss.backward()
        self.optimizer.step()

        # Clip weights of discriminator
        for p in self.parameters():
            p.data.clamp_(-0.01, 0.01)

        return loss.item()