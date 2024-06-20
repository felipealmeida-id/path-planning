from torch import ones
from torch.nn import BCELoss,Module
from env_parser import Env

class CustomLoss(Module):

    def __init__(self, evaluations, eval_weight, regular_weight):
        super(CustomLoss, self).__init__()
        self.evaluations = evaluations
        self.eval_weight = eval_weight
        self.regular_weight = regular_weight

    def adjust_weights(self,eval_weight:float,regular_weight:float):
        self.eval_weight = eval_weight
        self.regular_weight = regular_weight

    def set_evaluations(self,evaluations):
        self.evaluations = evaluations

    def forward(self, predictions, targets):
        env = Env.get_instance()
        loss_fun_eval = BCELoss()
        loss_fun = BCELoss()
        reg = self.regular_weight * loss_fun(predictions, targets)
        evaluation = self.eval_weight * loss_fun_eval(self.evaluations,ones(self.evaluations.size(0)).to(env.DEVICE))
        total = reg + evaluation
        return total
