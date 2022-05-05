import torch

class GradRevFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(self,x):
        return x.view_as(x)
    def backward(self,grad):
        lamb=1.0
        return -lamb*grad
    
class GradRevLayer(torch.nn.Module):
    def __init__(self, lamb=1):
        super().__init__()
        self.lamb = lamb
        
    def forward(self,x):
        return GradRevFunction.apply(x)