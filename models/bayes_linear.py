import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kfac_utils import *
from torch.distributions.multivariate_normal import MultivariateNormal


class BayesianLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 init_p: tuple[torch.Tensor, torch.Tensor],
                 init_q: tuple[torch.Tensor, torch.Tensor],
                 id: int,
                 kfac=False,
                 regularization=1e-5,
                 init_value=1e-2):
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

        self.in_features = in_features
        self.out_features = out_features

        # init prior
        self.p_weight_mu = nn.Parameter(init_p["weight"], requires_grad=False)
        self.p_bias_mu = nn.Parameter(init_p["bias"], requires_grad=False)

        # weight means (mu) and log-std (log_sigma)
        self.q_weight_mu = nn.Parameter(init_q["weight"])  
        
        # bias means (mu) and log-std (log_sigma)
        self.q_bias_mu = nn.Parameter(init_q["bias"]) 

        # for use in optimizer. will store gradients and A, G for momentum updates
        self.id = id

        if kfac:
            # F \approx A \otimes G
            self._A = init_value * torch.eye(self.in_features) + regularization * torch.eye(self.in_features)
            self._G = init_value * torch.eye(self.out_features) + regularization * torch.eye(self.out_features)
            self.q_bias_cov = init_value * torch.eye(self.out_features) + regularization * torch.eye(self.out_features)
        else:
            self.q_weight_log_sigma = nn.Parameter(0.5 * torch.log(torch.abs(self.q_weight_mu))) 
            self.q_bias_log_sigma = nn.Parameter(0.5 * torch.log(torch.abs(self.q_bias_mu)))  

    @staticmethod
    def kl_normal_diag(p_mu, p_sigma, q_mu, q_sigma):
        """
        Compute KL divergence between two normal distributions.
            
        Inputs: p is the posterior, q is the prior
        """
        return torch.sum(
            torch.log(p_sigma) - torch.log(q_sigma) + (q_sigma**2 + (p_mu - q_mu)**2) / (2 * p_sigma**2) - 0.5
        )

    def forward(self, x, p_log_sigma, epsilon = 1e-5):
        kl, outputs = None, None
        if self.kfac:
            # batch size x input_dim
            activations = x.clone().detach().double()
            # update kfactors using activations from the previous layer
            self._A = activations.T @ activations / activations.size(0) + epsilon * torch.eye(activations.shape[1], device=activations.device)

            p_sigma = torch.exp(p_log_sigma)
            weight = sample_from_kron_dist(self.q_weight_mu, self._A, self._G)
            weight = weight.view(self.out_features, self.in_features)
            
            bias = MultivariateNormal(loc=self.q_bias_mu, covariance_matrix=self.q_bias_cov).rsample().double()

            kl_weight = self.kl_divergence_kfac_weight(self.p_weight_mu, p_sigma, self.q_weight_mu, self._A, self._G)
            kl_bias = self.kl_divergence_kfac_bias(self.p_bias_mu, p_sigma, self.q_bias_mu, self.q_bias_cov)
            #print(kl_weight, kl_bias)
            kl = kl_weight + kl_bias
        
            outputs = F.linear(x.double(), weight, bias)


        else:
            p_sigma = torch.exp(p_log_sigma)
            q_weight_sigma = torch.exp(self.q_weight_log_sigma)
            q_bias_sigma = torch.exp(self.q_bias_log_sigma)

            weight = self.q_weight_mu + q_weight_sigma * torch.randn_like(q_weight_sigma)
            bias = self.q_bias_mu + q_bias_sigma * torch.randn_like(q_bias_sigma)

            kl_weight = self.kl_normal_diag(self.p_weight_mu, p_sigma, self.q_weight_mu, q_weight_sigma)
            kl_bias = self.kl_normal_diag(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
            kl = kl_weight + kl_bias

            outputs = F.linear(x, weight, bias)

        return outputs, kl
    
    def kl_divergence(self, p_log_sigma=None, A=None, G=None) -> float:
        """Returns KL divergence between prior and posterior distributions.

        Args:
            p_log_sigma (float): log std of prior distribution
            A, G: kronecker factors for prior

        Returns:
            float: KL divergence
        """
        if not self.kfac:
            p_sigma = torch.exp(p_log_sigma)
            q_weight_sigma = torch.exp(self.q_weight_log_sigma)
            q_bias_sigma = torch.exp(self.q_bias_log_sigma)
            
            kl_weight = self.kl_normal_diag(self.p_weight_mu, p_sigma, self.q_weight_mu, q_weight_sigma)
            kl_bias = self.kl_normal_diag(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
            kl = kl_weight + kl_bias
        else:
            # prior 
            inv_A = compute_inv(A)[1]
            inv_G = compute_inv(G)[1]
            prior_cov = torch.kron(inv_A, inv_G)

            q_weight_cov = torch.kron(self._A, self._G)
            kl_weight = self.kl_divergence_kfac_weight(self.p_weight_mu, prior_cov, self.q_weight_mu, q_weight_cov)
            # TODO: I'm not sure if prior cov should be the same for bias
            kl_bias = self.kl_divergence_kfac_bias(self.p_bias_mu, prior_cov, self.q_bias_mu, self.q_bias_cov)
            print(kl_weight, kl_bias)
            kl = kl_weight + kl_bias
        
        return kl
    
    def kl_divergence_kfac_weight(self, p_weight_mu, p_sigma, q_weight_mu, A, G, epsilon=1e-5):
        """
        Compute the KL divergence between a diagonal Gaussian prior and a Kronecker-factored Gaussian posterior.

        Args:
            p_weight_mu (torch.Tensor): Mean vector of the diagonal prior (m * n,).
            p_sigma (float): Standard deviation of the diagonal prior (scalar).
            q_weight_mu (torch.Tensor): Mean vector of the KFAC posterior (m * n,).
            A (torch.Tensor): Activation covariance matrix of the posterior (n x n).
            G (torch.Tensor): Gradient covariance matrix of the posterior (m x m).

        Returns:
            float: KL divergence D_KL(prior || posterior).
        """
        # Dimensions
        n = A.shape[0]  # Input size
        m = G.shape[0]  # Output size
        A = A + epsilon * torch.eye(A.shape[0], device=A.device)
        G = G + epsilon * torch.eye(G.shape[0], device=G.device)

        # Compute log determinants
        log_det_A = torch.logdet(A)
        log_det_G = torch.logdet(G)
        log_det_prior = n * torch.log(p_sigma**2)  # Diagonal prior covariance

        # Compute trace terms
        trace_A = torch.trace(A)  # Tr(A)
        trace_G = torch.trace(G)  # Tr(G)
        trace_term = trace_A * trace_G / p_sigma**2

        # Compute mean difference term
        delta_mu = q_weight_mu - p_weight_mu
        delta_mu_reshaped = delta_mu.view(m, n).double()  # Reshape to (m, n)

        inv_A = torch.linalg.inv(A).double()
        inv_G = torch.linalg.inv(G).double()

        quadratic_term = torch.trace(inv_G @ delta_mu_reshaped @ inv_A @ delta_mu_reshaped.T)

        # KL divergence formula
        kl = 0.5 * (
            (log_det_A + n * log_det_G - log_det_prior - m * n) + trace_term + quadratic_term
        )

        return kl
    
    def kl_divergence_kfac_bias(self, p_bias_mu, p_sigma, q_bias_mu, q_bias_cov):
        """
        Compute the KL divergence between a diagonal Gaussian prior and a full covariance Gaussian posterior for biases.

        Args:
            p_bias_mu (torch.Tensor): Mean vector of the diagonal prior (m,).
            p_sigma (float): Standard deviation of the diagonal prior (scalar).
            q_bias_mu (torch.Tensor): Mean vector of the posterior (m,).
            q_bias_cov (torch.Tensor): Covariance matrix of the posterior (m x m).

        Returns:
            float: KL divergence D_KL(prior || posterior).
        """
        # Dimensions
        m = p_bias_mu.shape[0]  # Number of biases

        # Compute log determinants
        log_det_q = torch.logdet(q_bias_cov)
        log_det_prior = m * torch.log(p_sigma**2)  # Diagonal prior covariance

        # Compute trace term
        trace_term = torch.trace(q_bias_cov) / p_sigma**2

        # Compute mean difference term
        delta_mu = q_bias_mu - p_bias_mu
        quadratic_term = delta_mu.T @ torch.linalg.inv(q_bias_cov) @ delta_mu / p_sigma**2

        # KL divergence formula
        kl = 0.5 * (
            log_det_q - log_det_prior - m + trace_term + quadratic_term
        )

        return kl

