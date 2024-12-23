# @title Imports
import torch
import torch.optim as optim


from models.bayes_linear import BayesianLinear

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
            if isinstance(layer, BayesianLinear):
                for name, param in layer.named_parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    
                    # only condition weights?
                    if name[2] == 'w': # weights
                        #print(torch.diag(2 * torch.exp(layer.q_bias_log_sigma)))
                        param.data -= self.defaults['lr'] * torch.diag(2 * torch.exp(layer.q_weight_log_sigma)) @ grad
                    elif name[2] == 'b': # biases
                        #print(self.defaults['lr'] * torch.diag(2 * torch.exp(layer.q_bias_log_sigma)))
                        param.data -= self.defaults['lr'] * torch.diag(2 * torch.exp(layer.q_bias_log_sigma)) @ grad
                    else:
                        raise ValueError(f"Unknown parameter name: {name}")

        self.model.p_log_sigma.data -= self.defaults['lr'] * self.model.p_log_sigma.grad

        return loss