# @title Imports
import torch
import torch.optim as optim


from models.bayes_nondiag import BayesNonDiag

device = ''

class DiagHessianOptimizer(optim.Optimizer):
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
            if isinstance(layer, BayesNonDiag):
                for name, param in layer.named_parameters():
                    grad = param.grad
                    if grad is None:
                        continue

                    if name[2] == 'w': # weights
                        param.data -= self.defaults['lr'] * torch.diag(2 * torch.exp(layer.q_weight_log_cov)) @ grad
                    elif name[2] == 'b': # biases
                        param.data -= self.defaults['lr'] * grad
                    else:
                        raise ValueError(f"Unknown parameter name: {name}")

        self.model.p_log_cov.data -= self.defaults['lr'] * self.model.p_log_cov.grad

        return loss