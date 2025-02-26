import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kfac_utils import *
from utils.sampling import *


class BayesianLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 init_p: tuple[torch.Tensor, torch.Tensor],
                 init_q: tuple[torch.Tensor, torch.Tensor],
                 approx="diagonal",
                 init_value=1e-2, 
                 precision="float32"):
        """
        Args:
            in_features (int): Input feature size
            out_features (int): Output feature size
            init_p (tuple[torch.Tensor, torch.Tensor]): Prior initial weight and bias means
            init_q (tuple[torch.Tensor, torch.Tensor]): Posterior initial weight and bias means
            cat (string): kfac, noisy-kfac (kfactored posterior), or diagonal
        """
        super(BayesianLinear, self).__init__()
        assert "weight" in init_p and "bias" in init_p, "Prior must contain both weight and bias"
        assert "weight" in init_q and "bias" in init_q, "Posterior must contain both weight and bias"

        assert init_p["weight"].shape == (out_features, in_features), "Prior weight shape mismatch"
        assert init_p["bias"].shape == (out_features,), "Prior bias shape mismatch"
        
        assert init_q["weight"].shape == (out_features, in_features), "Posterior weight shape mismatch"
        assert init_q["bias"].shape == (out_features,), "Posterior bias shape mismatch"

        self.approx = approx

        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision

        self.p_mu = nn.Parameter(
                torch.cat((init_p["weight"].to(getattr(torch, precision)), init_p['bias'].unsqueeze(1).to(getattr(torch, precision))), dim=1), 
                requires_grad=False
            )  
        self.q_bias_mu = None

        self.training = True
        self.bias = None # temporary hack because this is checked by the optimizer

        # TODO: does not currently work
        if self.approx == 'noisy-kfac':
            # F \approx A \otimes G
            self._A = init_value * torch.eye(self.in_features + 1, dtype=getattr(torch, precision))  #+ regularization * torch.eye(self.in_features)
            self._G = init_value * torch.eye(self.out_features, dtype=getattr(torch, precision)) #+ regularization * torch.eye(self.out_features)\

            self.A_inv = 1/init_value * torch.eye(self.in_features + 1, dtype=getattr(torch, precision))
            self.G_inv = 1/init_value * torch.eye(self.out_features, dtype=getattr(torch, precision))
            # initialize weights as MLP weights (we will compute gradients wrt weights)
            self.weights =  nn.Parameter(init_q["weight"].to(torch.float64))
            self.q_mu = nn.Parameter(torch.cat((init_q["weight"].to(getattr(torch, precision)), 
                                                init_q['bias'].unsqueeze(1).to(getattr(torch, precision))), dim=1), 
                                     requires_grad=False
                                     ) 
            self.dG, self.dA = None, None
            self.Q_G, self.Q_A = None, None
        
        elif self.approx == 'kfac':
            # we include the bias term in A
            self._A = init_value * torch.eye(self.in_features + 1)
            self._G = init_value * torch.eye(self.out_features)
            self.q_mu = nn.Parameter(torch.cat((init_q["weight"].to(getattr(torch, precision)), 
                                                init_q['bias'].unsqueeze(1).to(getattr(torch, precision))), dim=1)) 
            

        else:
            self.q_mu = nn.Parameter(torch.cat((init_q["weight"], init_q['bias'].unsqueeze(1)), dim=1)) 
            self.q_log_sigma = nn.Parameter(torch.cat((0.5 * torch.log(torch.abs(init_q["weight"].to(getattr(torch, precision)))), \
                                                   0.5 * torch.log(torch.abs(init_q["bias"])).unsqueeze(1).to(getattr(torch, precision))), dim=1)) 
            self.q_bias_mu = None
            


    def forward(self, x, p_log_sigma, flag, num_samples=5):
        if self.approx == 'noisy-kfac':
            x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1).type(getattr(torch, self.precision))
            # prior std
            p_sigma = torch.exp(p_log_sigma)

            kl = self.kl_divergence_kfac(self.p_mu, p_sigma, self.q_mu, flag)

            # sample the weights from MN(q_mu, lambda/N A^{-1}, G^{-1})
            self.weights.data = current_sampling(self.q_mu, self.A_inv, self.G_inv).view(self.out_features, self.in_features + 1)
            outputs = F.linear(x, self.weights, torch.zeros(self.out_features).type(getattr(torch, self.precision)))
        
        # kfactored posterior approximation
        elif self.approx == 'kfac':
            # append a column of 1's for bias term
            x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)
            # prior std
            p_sigma = torch.exp(p_log_sigma)
   
            #weights = 0
            #for _ in range(num_samples):
            #   weights += 1/num_samples * sample_from_kron_dist_fast(self.q_mu, self._A, self._G).view(self.out_features, self.in_features + 1)
            #print("PLOGSIGMA", p_log_sigma)

            # compute KL between kfactored posterior and diagonal prior
            kl = self.kl_divergence_kfac(self.p_mu, p_sigma, self.q_mu, flag)
            
            #weights = sample_from_kron_dist_fast(self.q_mu, self._A, self._G).view(self.out_features, self.in_features + 1)
            # outputs = F.linear(x, weights, torch.zeros(self.out_features))
            # sample activations instead of weights (Kingma et al, 2015)
            outputs = 0
            for _ in range(num_samples):
                outputs += 1/num_samples * sample_activations_kron_fast(self.q_mu, x, self._A, self._G)

        # diagonal approximation
        else:
            # append a column of 1's for bias term
            x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)
            
            p_sigma = torch.exp(p_log_sigma)
            q_sigma = torch.exp(self.q_log_sigma)

            weights = self.q_mu + q_sigma * torch.randn_like(q_sigma)
            kl = self.kl_normal_diag(self.p_mu, p_sigma, self.q_mu, q_sigma)

            outputs = F.linear(x, weights, torch.zeros(self.out_features).type(getattr(torch, self.precision)))

        return outputs, kl
    
    def kl_divergence(self, p_log_sigma=None, flag="train") -> float:
        """Returns KL divergence between prior and posterior distributions.

        Args:
            p_log_sigma (float): log std of prior distribution
            A, G: kronecker factors for prior

        Returns:
            float: KL divergence
        """
        kl = None
        p_sigma = torch.exp(p_log_sigma)
        
        if self.approx == 'diagonal':
            q_sigma = torch.exp(self.q_log_sigma)
            kl = self.kl_normal_diag(self.p_mu, p_sigma, self.q_mu, q_sigma)
        else:
            kl = self.kl_divergence_kfac(self.p_mu, p_sigma, self.q_mu, flag)
        
        return kl
    

    def kl_divergence_kfac(self, p_mu, p_sigma, q_weight_mu, flag, epsilon=1e-1):
        #NOTE: make sure the eps aligns with clamping param in kfac/noisy-kfac
        """
        Compute the KL divergence between a diagonal Gaussian prior and a Kronecker-factored Gaussian posterior.

        Args:
            p_weight_mu (torch.Tensor): Mean vector of the diagonal prior (shape m*n,).
            p_sigma (float): Standard deviation of the diagonal prior (scalar).
            q_weight_mu (torch.Tensor): Mean vector of the KFAC posterior (shape m*n,).
            epsilon (float): Small value for numerical stability (defaults to 1e-2).

        Returns:
            torch.Tensor: KL divergence D_KL(prior || posterior).
        """
        # Retrieve the KFAC blocks from this class
        A, G = self._A, self._G
        n, m = A.shape[0], G.shape[0]

        # values have not yet been computed by the optimizer
        if self.Q_A is None:
            A = A + torch.eye(n, device=A.device) * epsilon
            G = G + torch.eye(m, device=G.device) * epsilon
            
            # eigendecompositions
            dA, Q_A = torch.linalg.eigh(A)
            dA = torch.clamp(dA, min=epsilon) 

            inv_dA = 1.0 / dA
            A_inv = Q_A @ torch.diag(inv_dA + epsilon) @ Q_A.T
            
            dG, Q_G = torch.linalg.eigh(G) 
            dG = torch.clamp(dG, min=epsilon)
            
            # Inverse of G  = Q_G @ diag(1/dG) @ Q_G^T
            inv_dG = 1.0 / dG
            G_inv = Q_G @ torch.diag(inv_dG + epsilon) @ Q_G.T

            log_det_A = torch.sum(torch.log(dA))
            log_det_G = torch.sum(torch.log(dG))

        else:
            log_det_A = torch.sum(torch.log(self.dA))
            log_det_G = torch.sum(torch.log(self.dG)) 
            A_inv, G_inv = self.A_inv, self.G_inv         
        
        
        log_det_prior = n * m * torch.log(p_sigma**2)
        
        # trace_term = trace(A) * trace(G) / p_sigma^2
        trace_term = torch.trace(A_inv) * torch.trace(G_inv) * 1.0/(p_sigma**2)
        
        diff = q_weight_mu - p_mu
        quad_term = (1.0 / (p_sigma**2)) * torch.sum(diff**2)

        kl = 0.5 * (
            (m * log_det_A + n * log_det_G + log_det_prior - m * n)
            + trace_term
            + quad_term
        )
        
        if kl < 0:
            kl = torch.tensor(0)
    
        return kl
    
    def kl_normal_diag(self, p_mu, p_sigma, q_mu, q_sigma):
        """
        Compute KL divergence between two normal distributions.
            
        Inputs: p is the posterior, q is the prior
        """
        return torch.sum(
            torch.log(p_sigma) - torch.log(q_sigma) + (q_sigma**2 + (p_mu - q_mu)**2) / (2 * p_sigma**2) - 0.5
        )