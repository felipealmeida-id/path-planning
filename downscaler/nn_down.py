from torch import load
from torch.nn import Module,Sequential,Linear,MSELoss
from torch.optim import Optimizer, Adam
from torch.nn.modules.loss import _Loss


class DownscalerNN(Module):
    optimizer:Optimizer
    loss_fun:_Loss = MSELoss()

    def __init__(self):
        from env_parser import Env
        env = Env.get_instance()
        super(DownscalerNN, self).__init__()
        self.input_size = env.UAV_AMOUNT * env.HR_TOTAL_TIME
        self.output_size = env.UAV_AMOUNT * env.TOTAL_TIME
        self.layer_chain = Sequential(
            Linear(self.input_size,self.output_size)
        )
        self.optimizer = Adam(self.parameters(), lr=env.D_LEARN_RATE)

    def forward(self,x):
        from env_parser import Env
        env = Env.get_instance()
        x = x.view(-1,800)
        return self.layer_chain(x).view(-1, env.UAV_AMOUNT, env.TOTAL_TIME)
    
    def load_pretrained_model(self):
        return
        from env_parser import Env
        env = Env.get_instance()
        self.load_state_dict(load(env.TRAINED_DOWNSCALER_PATH))

    def custom_train(self,data,target):
        self.train()
        self.optimizer.zero_grad()
        output = self(data)
        output = output.view(400)
        loss = self.loss_fun(output,target)
        loss.backward()
        self.optimizer.step()
