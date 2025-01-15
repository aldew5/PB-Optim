import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayes_linear import BayesianLinear

import math


class BayesianNN(nn.Module):
    def __init__(self,
                 init_p_weights: list[tuple[torch.Tensor, torch.Tensor]],
                 init_q_weights: list[tuple[torch.Tensor, torch.Tensor]],
                 in_features: int = 784,
                 out_features: int = 1,
                 hidden_features: int = 300,
                 p_log_sigma: float = -6,
                 approx="diagonal"):
        """Bayesian Neural Network (BNN) model w/ 3 layers.

        Args:
            init_p_weights (list[tuple[torch.Tensor, torch.Tensor]]): Initial prior weights and biases for each layer.
            init_q_weights (list[tuple[torch.Tensor, torch.Tensor]]): Initial posterior weights and biases for each layer.
            in_features (int, optional): Input feature size. Defaults to 784.
            out_features (int, optional): Output feature size. Defaults to 1.
            hidden_features (int, optional): Hidden layer size. Defaults to 300.
            p_log_sigma (int, optional): Prior log std. Defaults to -6.
            kfac (bool, optional): KFAC optimization. Defaults to False.
        """
        super(BayesianNN, self).__init__()

        assert len(init_p_weights) == 3, f"Number of prior weights must be 3, got {len(init_p_weights)}"
        assert len(init_q_weights) == 3, f"Number of posterior weights must be 3, got {len(init_q_weights)}"
        
        self.approx = approx
        self.p_log_sigma = nn.Parameter(torch.tensor(p_log_sigma, dtype=torch.float32))
        

        self.in_features = in_features
        self.out_features = out_features
        
        
        self.bl1 = BayesianLinear(in_features, hidden_features, init_p_weights[0], init_q_weights[0], 1, approx=approx)
        self.bl2 = BayesianLinear(hidden_features, hidden_features, init_p_weights[1], init_q_weights[1], 2, approx=approx)
        self.bl3 = BayesianLinear(hidden_features, 1, init_p_weights[2], init_q_weights[2], 3, approx=approx)

        self.layers = [self.bl1, self.bl2, self.bl3]


    def forward(self, x, p_log_sigma=None):
        if p_log_sigma is None:
            p_log_sigma = self.p_log_sigma
            
        x = x.view(-1, self.in_features)

        x, kl1 = self.bl1(x, p_log_sigma)
        x = F.relu(x)

        x, kl2 = self.bl2(x, p_log_sigma)
        x = F.relu(x)

        x, kl3 = self.bl3(x, p_log_sigma)
        
        #print(x, kl1, kl2, kl3)
        return x, kl1 + kl2 + kl3, p_log_sigma
    
    def kl_divergence(self, p_log_sigma=None):
        if p_log_sigma is None and not self.kfac:
            p_log_sigma = self.p_log_sigma

        
        return self.bl1.kl_divergence(p_log_sigma=p_log_sigma) + self.bl2.kl_divergence(p_log_sigma=p_log_sigma) +\
                self.bl3.kl_divergence(p_log_sigma=p_log_sigma)
     