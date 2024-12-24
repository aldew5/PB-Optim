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
    def __init__(self, model, lr=0.01, damping=1e-1, ema_decay=0.8, momentum=0.8):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, damping=damping, ema_decay=ema_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.model = model
        self.state = {}

    def _smooth_update(self, old, new, decay):
        """Apply an exponential moving average to smooth updates."""
        if old is None:
            return new
        return decay * old + (1 - decay) * new

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for layer in self.model.modules():
            if isinstance(layer, BayesianLinear):
                state = self.state.setdefault(layer, {})
                A_old = state.get("A")
                G_old = state.get("G")
                A, G = layer._A, layer._G

                if A is not None and G is not None:
                    A_new = self._smooth_update(
                    A_old.detach() if A_old is not None else None,
                    A.detach(),
                    self.defaults['ema_decay']
                    )
                    G_new = self._smooth_update(
                    G_old.detach() if G_old is not None else None,
                    G.detach(),
                    self.defaults['ema_decay'])

                    state["A"] = A_new
                    state["G"] = G_new
                    A_inv = torch.inverse(A_new + self.defaults['damping'] * torch.eye(A_new.size(0)).to(A_new.device))
                    G_inv = torch.inverse(G_new + self.defaults['damping'] * torch.eye(G_new.size(0)).to(G_new.device))
                else:
                    continue
                
                for name, param in layer.named_parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    
                    if name[2] == 'w': # weights
                        natural_grad = kron_mv(A_inv, G_inv, grad.view(-1))
                        state = self.state.setdefault(param, {})
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(natural_grad)

                        buf = state["momentum_buffer"]
                        buf.mul_(self.defaults['momentum']).add_(natural_grad)
                        param.data -= self.defaults['lr'] * buf.view_as(param)

                    elif name[2] == 'b': # biases
                        state = self.state.setdefault(param, {})
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(grad)

                        buf = state["momentum_buffer"]
                        buf.mul_(self.defaults['momentum']).add_(grad)
                        param.data -= self.defaults['lr'] * buf
                    else:
                        raise ValueError(f"Unknown parameter name: {name}")
                    
                  
        self.model.p_log_sigma.data -= self.defaults['lr'] * self.model.p_log_sigma.grad

        return loss