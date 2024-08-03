from torch import load
from torch.nn import Module,Sequential,Linear,ReLU
from env_parser import Env

class NeuralDownscaler(Module):
    def __init__(self):
        env = Env.get_instance()
        super(NeuralDownscaler, self).__init__()
        self.seq = Sequential(
            Linear(2 * env.HR_TOTAL_TIME,1024),
            # ReLU(),
            Linear(1024,2 * env.TOTAL_TIME)
        )

    def forward(self, x):
        env = Env.get_instance()
        x = x.view(-1, 2 * env.HR_TOTAL_TIME)
        x = self.seq(x)
        x = x.view(-1,env.TOTAL_TIME,2)
        return x

    def load_pretrained_model(self):
        from env_parser import Env
        env = Env.get_instance()
        pretrained_weights = load(env.TRAINED_DOWNSCALER_PATH)
        self.load_state_dict(pretrained_weights)
        for param in self.parameters():
            param.requires_grad = False

    