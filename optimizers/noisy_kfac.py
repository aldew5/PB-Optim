import torch
import torch.optim as optim
from models.bayes_linear import BayesianLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NoisyKFAC(optim.Optimizer):
    def __init__(self, 
            model,
            lr=0.01, 
            damping=1e-1, 
            beta1: float = 0.9,
            beta2: float = 0.99999,
            weight_decay: float = 1e-4,
            ess: int = 1e6,
            lam: int = 0.5,
            batch_size=100
        ):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, 
                        damping=damping, 
                        beta1=beta1,
                        beta2=beta2, 
                        weight_decay=weight_decay,
                        ess=ess, 
                        lam=lam, 
                        batch_size=batch_size)
        super().__init__(params, defaults)
        self.model = model

        # only for kfac-enabled BNNs
        assert(model.kfac)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for layer in self.model.layers:
            for name, param in layer.named_parameters():
                if name == 'weights':
                    # update the layer posterior mean
                    #print("param", param)
                    #print(param.grad)
                    with torch.no_grad():
                        V = param.grad - self.defaults['lam'] * layer.weights
                    layer.q_weight_mu += self.defaults['lr'] * layer.G_inv @ V @ layer.A_inv
                else:
                    # simple gradient descent on other parameters (bias mean)
                    if param.grad is not None:
                        param.data -= self.defaults['lr'] * param.grad

        return loss
    