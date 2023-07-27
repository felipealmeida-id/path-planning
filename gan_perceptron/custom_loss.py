import torch
import torch.nn as nn
from env_parser import Env

class CustomLoss(nn.Module):

    def __init__(self, evaluations, eval_weight, regular_weight):
        super(CustomLoss, self).__init__()
        self.evaluations = evaluations
        self.eval_weight = eval_weight
        self.regular_weight = regular_weight

    def adjust_weights(self,eval_weight:float,regular_weight:float):
        self.evalWeight = eval_weight
        self.regular_weight = regular_weight

    def set_evaluations(self,evaluations):
        self.evaluations = evaluations

    def forward(self, predictions, targets):
        env = Env.get_instance()
        reg = self.regular_weight * nn.BCELoss()(predictions, targets)
        evaluation = self.eval_weight * nn.BCELoss()(self.evaluations,
                                                    torch.ones(self.evaluations.size(0)).to(env.DEVICE))
        total = reg + evaluation
        return total
