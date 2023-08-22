from torch.nn import Module

class Detupler(Module):
    def forward(self,x):
        tensor, _ = x
        return tensor