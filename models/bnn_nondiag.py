import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bayes_nondiag import BayesNonDiag


class BayesianNonDiag(nn.Module):
    def __init__(self,
                 init_p_weights: list[tuple[torch.Tensor, torch.Tensor]],
                 init_q_weights: list[tuple[torch.Tensor, torch.Tensor]],
                 in_features: int = 784,
                 out_features: int = 1,
                 hidden_features: int = 300,
                 p_log_sigma: float = -6,
                 kfac=False):
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
        super(BayesianNonDiag, self).__init__()

        assert len(init_p_weights) == 3, f"Number of prior weights must be 3, got {len(init_p_weights)}"
        assert len(init_q_weights) == 3, f"Number of posterior weights must be 3, got {len(init_q_weights)}"
        
        self.kfac = kfac

        self.in_features = in_features
        self.out_features = out_features
        
        # proir cov
        self.p_log_cov = nn.Parameter(torch.normal(mean=torch.exp(torch.tensor(p_log_sigma)), std=0.5))

        self.bl1 = BayesNonDiag(in_features, hidden_features, init_p_weights[0], init_q_weights[0], kfac=kfac)
        self.bl2 = BayesNonDiag(hidden_features, hidden_features, init_p_weights[1], init_q_weights[1], kfac=kfac)
        self.bl3 = BayesNonDiag(hidden_features, 1, init_p_weights[2], init_q_weights[2], kfac=kfac)

    def forward(self, x, p_log_sigma=None):
        if p_log_sigma is None:
            p_log_sigma =self.p_log_cov
            
        x = x.view(-1, self.in_features)

        x, kl1 = self.bl1(x, p_log_sigma)
        x = F.relu(x)

        x, kl2 = self.bl2(x, p_log_sigma)
        x = F.relu(x)

        x, kl3 = self.bl3(x, p_log_sigma)

        return x, kl1 + kl2 + kl3, p_log_sigma
    
    def kl_divergence(self, p_log_sigma=None):
        if p_log_sigma is None:
            p_log_sigma = self.p_log_cov
            
        return self.bl1.kl_divergence(p_log_sigma) + self.bl2.kl_divergence(p_log_sigma) + self.bl3.kl_divergence(p_log_sigma)