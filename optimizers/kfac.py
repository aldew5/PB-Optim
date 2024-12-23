# @title Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


from models.bayes_linear import BayesianLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def kron_mv(A, G, v):
    # Efficient matrix-vector multiplication using the Kronecker product
    m, n = A.shape[0], G.shape[0]
    v = v.view(n, m)
    return (G @ v) @ A.t()

class KFACOptimizer(optim.Optimizer):
    def __init__(self, model, lr=0.01, damping=1e-1):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, damping=damping)
        super().__init__(params, defaults)
        self.model = model

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for layer in self.model.modules():
            if isinstance(layer, BayesianLinear):
                for name, param in layer.named_parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    
                    if name[2] == 'w': # weights
                        A, G = layer._A, layer._G
                        if A is None or G is None:
                            continue

                        A_inv = torch.inverse(A + self.defaults['damping'] * torch.eye(A.size(0)).to(device))
                        G_inv = torch.inverse(G + self.defaults['damping'] * torch.eye(G.size(0)).to(device))
                        natural_grad = kron_mv(A_inv, G_inv, grad.view(-1))

                        param.data -= self.defaults['lr'] * natural_grad
                    elif name[2] == 'b': # biases
                        param.data -= self.defaults['lr'] * grad
                    else:
                        raise ValueError(f"Unknown parameter name: {name}")
                    
                  
        self.model.p_log_sigma.data -= self.defaults['lr'] * self.model.p_log_sigma.grad

        return loss