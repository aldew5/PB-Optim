import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init_cov import init_cov


class BayesNonDiag(nn.Module):
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
        super(BayesNonDiag, self).__init__()
        assert "weight" in init_p and "bias" in init_p, "Prior must contain both weight and bias"
        assert "weight" in init_q and "bias" in init_q, "Posterior must contain both weight and bias"

        assert init_p["weight"].shape == (out_features, in_features), "Prior weight shape mismatch"
        assert init_p["bias"].shape == (out_features,), "Prior bias shape mismatch"
        
        assert init_q["weight"].shape == (out_features, in_features), "Posterior weight shape mismatch"
        assert init_q["bias"].shape == (out_features,), "Posterior bias shape mismatch"

        self.kfac = kfac
        if kfac:
            self._A = None
            self._G = None

        self.in_features = in_features
        self.out_features = out_features

        # init prior
        self.p_weight_mu = init_p["weight"]
        self.p_bias_mu = init_p["bias"]

        N = in_features * out_features

        # weight means (mu) and log-std (log_sigma)
        self.q_weight_mu = nn.Parameter(init_q["weight"])  
        self.q_weight_log_cov = nn.Parameter(init_cov(init_q["weight"], 0.5 * torch.abs(self.q_weight_mu)))
        #with torch.no_grad():
        #    self.q_weight_log_cov[torch.isnan(self.q_weight_log_cov)] = 0.001
        #    self.q_weight_log_cov[self.q_weight_log_cov < 0] = 0.001
        print("INIT", self.q_weight_log_cov)

        # bias means (mu) and log-std (log_sigma)
        self.q_bias_mu = nn.Parameter(init_q["bias"]) 
        self.q_bias_log_cov = nn.Parameter(torch.log(torch.normal(init_q["bias"], 0.5 * torch.abs(self.q_bias_mu))))
        #with torch.no_grad():
        #    self.q_bias_log_cov[torch.isnan(self.q_bias_log_cov)] = 0.001
        #    self.q_bias_log_cov[self.q_bias_log_cov < 0] = 0.001

    @staticmethod
    def kl_normal(p_mu, p_cov, q_mu, q_cov):
        """
        Compute KL divergence between two normal non-diagonal distributions.
            
        Inputs: p is the posterior, q is the prior
        """
        inv_q_cov = torch.linalg.inv(q_cov)

        return 0.5 * torch.log(torch.det(q_cov)/torch.det(p_cov)) - p_cov.shape(0) +\
            torch.trace(inv_q_cov * p_cov) + (q_mu - p_mu).T * inv_q_cov * (q_mu - p_mu)

    def forward(self, x, p_log_cov):
        if self.kfac:
            activations = x.clone().detach()  # Detach to avoid unnecessary gradients
            self._A = activations.t() @ activations / activations.size(0)

        # Using reparameterization trick (rsample)
        p_cov = torch.exp(p_log_cov)
        q_weight_cov = torch.exp(self.q_weight_log_cov)
        #print(q_weight_cov, self.q_weight_log_cov)
        q_bias_cov = torch.exp(self.q_bias_log_cov)

        q_weight_distr = torch.distributions.MultivariateNormal(loc=self.q_weight_mu, covariance_matrix=q_weight_cov)
        q_bias_distr = torch.distributions.MultivariateNormal(loc=self.q_bias_mu, covariance_matrix=q_bias_cov)
        weight = q_weight_distr.sample()
        bias = q_bias_distr.sample()

        kl_weight = self.kl_normal(self.p_weight_mu, p_cov, self.q_weight_mu, q_weight_cov)
        kl_bias = self.kl_normal(self.p_bias_mu, p_cov, self.q_bias_mu, q_bias_cov)
        kl = kl_weight + kl_bias

        outputs = F.linear(x, weight, bias)

        if self.kfac:
            self._G = outputs.t() @ outputs / outputs.size(0)

        #print("FORWRAD")
        return outputs, kl
    
    def kl_divergence(self, p_log_cov: float) -> float:
        """Returns KL divergence between prior and posterior distributions.

        Args:
            p_log_sigma (float): log std of prior distribution

        Returns:
            float: KL divergence
        """
        p_cov= torch.exp(p_log_cov)
        q_weight_cov = torch.exp(self.q_weight_log_cov)
        q_bias_cov = torch.exp(self.q_bias_log_cov)
        
        kl_weight = self.kl_normal(self.p_weight_mu, p_cov, self.q_weight_mu, q_weight_cov)
        kl_bias = self.kl_normal(self.p_bias_mu, p_cov, self.q_bias_mu, q_bias_cov)
        kl = kl_weight + kl_bias
        
        return kl
