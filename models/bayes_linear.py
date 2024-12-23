import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 init_p: tuple[torch.Tensor, torch.Tensor],
                 init_q: tuple[torch.Tensor, torch.Tensor],
                 kfac=False):
        """Bayesian Linear Layer.

        Args:
            in_features (int): Input feature size
            out_features (int): Output feature size
            init_p (tuple[torch.Tensor, torch.Tensor]): Prior initial weight and bias means
            init_q (tuple[torch.Tensor, torch.Tensor]): Posterior initial weight and bias means
            kfac (bool, optional): KFAC optimization. Defaults to False.
        """
        super(BayesianLinear, self).__init__()
        assert "weight" in init_p and "bias" in init_p, "Prior must contain both weight and bias"
        assert "weight" in init_q and "bias" in init_q, "Posterior must contain both weight and bias"

        assert init_p["weight"].shape == (out_features, in_features), "Prior weight shape mismatch"
        assert init_p["bias"].shape == (out_features,), "Prior bias shape mismatch"
        
        assert init_q["weight"].shape == (out_features, in_features), "Posterior weight shape mismatch"
        assert init_q["bias"].shape == (out_features,), "Posterior bias shape mismatch"

        self.kfac = kfac
        if kfac:
            # approx activation cov
            self._A = None
            # approx gradient cov
            self._G = None

        self.in_features = in_features
        self.out_features = out_features

        # init prior
        self.p_weight_mu = init_p["weight"]
        self.p_bias_mu = init_p["bias"]

        # weight means (mu) and log-std (log_sigma)
        self.q_weight_mu = nn.Parameter(init_q["weight"])  
        self.q_weight_log_sigma = nn.Parameter(0.5 * torch.log(torch.abs(self.q_weight_mu))) 

        # bias means (mu) and log-std (log_sigma)
        self.q_bias_mu = nn.Parameter(init_q["bias"]) 
        self.q_bias_log_sigma = nn.Parameter(0.5 * torch.log(torch.abs(self.q_bias_mu)))  

    @staticmethod
    def kl_normal(p_mu, p_sigma, q_mu, q_sigma):
        """
        Compute KL divergence between two normal distributions.
            
        Inputs: p is the posterior, q is the prior
        """
        return torch.sum(
            torch.log(p_sigma) - torch.log(q_sigma) + (q_sigma**2 + (p_mu - q_mu)**2) / (2 * p_sigma**2) - 0.5
        )

    def forward(self, x, p_log_sigma):
        if self.kfac:
            activations = x.clone().detach()  # Detach to avoid unnecessary gradients
            self._A = activations.T @ activations / activations.size(0)

        # Using reparameterization trick (rsample)
        p_sigma = torch.exp(p_log_sigma)
        q_weight_sigma = torch.exp(self.q_weight_log_sigma)
        q_bias_sigma = torch.exp(self.q_bias_log_sigma)

        weight = self.q_weight_mu + q_weight_sigma * torch.randn_like(q_weight_sigma)
        bias = self.q_bias_mu + q_bias_sigma * torch.randn_like(q_bias_sigma)

        kl_weight = self.kl_normal(self.p_weight_mu, p_sigma, self.q_weight_mu, q_weight_sigma)
        kl_bias = self.kl_normal(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
        kl = kl_weight + kl_bias

        outputs = F.linear(x, weight, bias)

        if self.kfac:
            self._G = outputs.T @ outputs / outputs.size(0)

        return outputs, kl
    
    def kl_divergence(self, p_log_sigma: float) -> float:
        """Returns KL divergence between prior and posterior distributions.

        Args:
            p_log_sigma (float): log std of prior distribution

        Returns:
            float: KL divergence
        """
        p_sigma = torch.exp(p_log_sigma)
        q_weight_sigma = torch.exp(self.q_weight_log_sigma)
        q_bias_sigma = torch.exp(self.q_bias_log_sigma)
        
        kl_weight = self.kl_normal(self.p_weight_mu, p_sigma, self.q_weight_mu, q_weight_sigma)
        kl_bias = self.kl_normal(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
        kl = kl_weight + kl_bias
        
        return kl
