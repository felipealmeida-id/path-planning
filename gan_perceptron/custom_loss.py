from torch import ones, zeros
from torch.nn import BCELoss, Module, MSELoss
from env_parser import Env


class CustomLoss(Module):
    def __init__(self, evaluations, eval_weight, regular_weight):
        super(CustomLoss, self).__init__()
        self.evaluations = evaluations
        self.eval_weight = eval_weight
        self.regular_weight = regular_weight

    def adjust_weights(self, eval_weight: float, regular_weight: float):
        self.eval_weight = eval_weight
        self.regular_weight = regular_weight

    def set_evaluations(self, evaluations):
        self.evaluations = evaluations

    def forward(self, predictions, targets):
        env = Env.get_instance()
        loss_fun = BCELoss()
        reg = loss_fun(predictions, targets)
        # evaluation = self.eval_weight * loss_fun(
        #     self.evaluations, ones(self.evaluations.size(0)).to(env.DEVICE)
        # )
        evaluation = MSELoss()(self.evaluations, zeros(self.evaluations.size(0)).to(env.DEVICE))
        print("evaluation", evaluation)
        total = reg + evaluation
        return total
