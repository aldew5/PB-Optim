import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kfac_utils import *
import math


class BayesianLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 init_p: tuple[torch.Tensor, torch.Tensor],
                 init_q: tuple[torch.Tensor, torch.Tensor],
                 id: int,
                 approx="diagonal",
                 init_value=1e-2,
                 lam: int = 1e-1,
                 damping= 1e-2):
        """Bayesian Linear Layer.

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

        # init prior
        self.p_weight_mu = nn.Parameter(init_p["weight"], requires_grad=False)
        self.p_bias_mu = nn.Parameter(init_p["bias"], requires_grad=False)
        

        # for use in optimizer. will store gradients and A, G for momentum updates
        self.id = id

        if self.approx == 'noisy-kfac':
            # F \approx A \otimes G
            self._A = (init_value * torch.eye(self.in_features)).double() #+ regularization * torch.eye(self.in_features)
            self._G = (init_value * torch.eye(self.out_features)).double() #+ regularization * torch.eye(self.out_features)
            self.A_inv, self.G_inv = None, None
            #self.q_bias_cov = nn.Parameter(init_value * torch.eye(self.out_features) + regularization * torch.eye(self.out_features))
            self.q_bias_log_sigma = nn.Parameter(0.5 * torch.log(torch.abs(self.q_bias_mu)))  
            self.q_weight_mu = init_q["weight"].detach().clone()
            # initialize weights as MLP weights 
            self.weights =  nn.Parameter(init_q["weight"])
            
            self.lam = lam
            # iteration number for updating kronecker factors
            self.k = 0
            self.damping = damping
            
        elif self.approx == 'kfac':
            # we include the bias term in A
            self._A = (init_value * torch.eye(self.in_features + 1))
            self._G = (init_value * torch.eye(self.out_features))
            self.A_inv, self.G_inv = None, None
            self.log_det_A, self.log_det_G = None, None

            #self.q_weight_mu = nn.Parameter(init_q["weight"])
            #self.q_bias_log_sigma = nn.Parameter(0.5 * torch.log(torch.abs(self.q_bias_mu)))    
            self.q_mu = nn.Parameter(torch.cat((init_q["weight"], init_q['bias'].unsqueeze(1)), dim=1)) 
            self.p_mu = nn.Parameter(torch.cat((init_p["weight"], init_p['bias'].unsqueeze(1)), dim=1), requires_grad=False)  
            self.q_bias_mu = None
            self.training = True
            self.prev_kl = None

        else:
            # weight means (mu) and log-std (log_sigma)
            self.q_weight_mu = nn.Parameter(init_q["weight"])  
            # bias means (mu) and log-std (log_sigma)
            self.q_bias_mu = nn.Parameter(init_q["bias"]) 

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

    def forward(self, x, p_log_sigma, flag, T_stats=20, beta_1=0.9):
        """
            params:
                - k: current iteration number
                - T_stats: frequency of updating A
        """
        kl, outputs = None, None
        if self.approx == 'noisy-kfac':
            batch_size = x.size(0)
            p_sigma = torch.exp(p_log_sigma)
            q_bias_sigma = torch.exp(self.q_bias_log_sigma)
            gamma_in = self.lam/batch_size

            # compute inverses on the first iteration
            if self.k == 0:
                self.A_inv, self.G_inv = self._noisy_inversion(p_log_sigma, batch_size)
            
            
            row_cov = self.G_inv + self.damping * torch.eye(self._G.size(0))
            col_cov = gamma_in * self.A_inv + self.damping * torch.eye(self._A.size(0))
            # sample weights from matrix-variate normal distr parameterized with kronecker layer factors
            with torch.no_grad():
                self.weights.data = nn.Parameter(sample_mvnd(self.q_weight_mu, row_cov, col_cov))

            bias = self.q_bias_mu + q_bias_sigma * torch.randn_like(q_bias_sigma)

            kl_weight = self.kl_divergence_kfac_weight(self.p_weight_mu, p_sigma, self.q_weight_mu, flag)
            kl_bias = self.kl_normal_diag(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
            kl = kl_weight + kl_bias

            outputs = F.linear(x.double(), self.weights.double(), bias.double())

            # update A every T_stats iterations
            if  self.k % T_stats == 0:
                # batch size x input_dim
                activations = x.clone().detach()
                # update kfactors using activations from the previous layer
                self._A = (1 - beta_1) * self._A  + beta_1 * activations.T @ activations / activations.size(0) 
                self.A_inv, self.G_inv = self._noisy_inversion(p_log_sigma, batch_size)
                
            self.k += 1
        
        elif self.approx == 'kfac':
            # append a column of 1's for bias term
            x = torch.cat([x, torch.ones(x.size(0), 1)], dim=1)
            # update A
            #activations = x.clone().detach()
            #self._A = activations.T @ activations / activations.size(0)

            p_sigma = torch.exp(p_log_sigma)
            #q_bias_sigma = torch.exp(self.q_bias_log_sigma)

            # use means for forward
            weights = self.q_mu

            #weights = torch.zeros_like(self.q_mu)
            weights = sample_from_kron_dist_fast(self.q_mu, self._A, self._G).view(self.out_features, self.in_features + 1)
            #print(weights.diagonal())
            #print(weights)
            #print('weights', weights)
            #bias = self.q_bias_mu #+ q_bias_sigma * torch.randn_like(q_bias_sigma)

            kl = self.kl_divergence_kfac_weight(self.p_mu, p_sigma, self.q_mu, flag)
            #kl_bias = self.kl_normal_diag(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
            #kl = kl_weight + kl_bias
            #tensor(
            outputs = F.linear(x, weights, torch.zeros(self.out_features))
            #print("WEIGHTS", weights.size())
            #outputs = x @ weights.T

        # diagonal approximation
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
    
    def kl_divergence(self, p_log_sigma=None, flag="train") -> float:
        """Returns KL divergence between prior and posterior distributions.

        Args:
            p_log_sigma (float): log std of prior distribution
            A, G: kronecker factors for prior

        Returns:
            float: KL divergence
        """
        kl = None
        if self.approx == 'diagonal':
            p_sigma = torch.exp(p_log_sigma)
            q_weight_sigma = torch.exp(self.q_weight_log_sigma)
            q_bias_sigma = torch.exp(self.q_bias_log_sigma)
            
            kl_weight = self.kl_normal_diag(self.p_weight_mu, p_sigma, self.q_weight_mu, q_weight_sigma)
            kl_bias = self.kl_normal_diag(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
            kl = kl_weight + kl_bias
        else:
            p_sigma = torch.exp(p_log_sigma)
            #q_bias_sigma = torch.exp(self.q_bias_log_sigma)

            kl = self.kl_divergence_kfac_weight(self.p_mu, p_sigma, self.q_mu, flag)
            #kl_bias = self.kl_normal_diag(self.p_bias_mu, p_sigma, self.q_bias_mu, q_bias_sigma)
            #kl = kl_weight + kl_bias
            #print(kl_weight, kl_bias) 
        
        return kl
    
    def kl_divergence_kfac_weight(self, p_mu, p_sigma, q_weight_mu, flag, epsilon=1e-3):
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

        trace_A = torch.trace(A)
        trace_G = torch.trace(G)
        A = A / (trace_A + epsilon)
        G = G / (trace_G + epsilon)
        
        # Add a small epsilon on the diagonal to ensure positive-definiteness
        A = A + torch.eye(n, device=A.device) * epsilon
        G = G + torch.eye(m, device=G.device) * epsilon
        
        # ----------------------------------------------------------------
        # 1) Eigendecomposition of A
        #    A = Q_A @ diag(dA) @ Q_A^T,   with dA = eigenvalues, Q_A = eigenvectors
        # ----------------------------------------------------------------
        dA, Q_A = torch.linalg.eigh(A)  # or torch.symeig(A, eigenvectors=True)
        dA = torch.clamp(dA, min=epsilon)  # Ensure eigenvalues are positive
        # Log-determinant of A = sum(log(dA))
        log_det_A = torch.sum(torch.log(dA))
        # Inverse of A  = Q_A @ diag(1/dA) @ Q_A^T
        inv_dA = 1.0 / dA
        A_inv = Q_A @ torch.diag(inv_dA) @ Q_A.T
        
        # ----------------------------------------------------------------
        # 2) Eigendecomposition of G
        #    G = Q_G @ diag(dG) @ Q_G^T
        # ----------------------------------------------------------------
        dG, Q_G = torch.linalg.eigh(G)  # or torch.symeig(G, eigenvectors=True)
        dG = torch.clamp(dG, min=epsilon)

        # Log-determinant of G = sum(log(dG))
        log_det_G = torch.sum(torch.log(dG))
        # Inverse of G  = Q_G @ diag(1/dG) @ Q_G^T
        inv_dG = 1.0 / dG
        G_inv = Q_G @ torch.diag(inv_dG) @ Q_G.T
        
        # ----------------------------------------------------------------
        # 3) Remaining terms for KL divergence
        # ----------------------------------------------------------------
        # log(det(prior)) = n * log(p_sigma^2)
        log_det_prior = n * m* torch.log(p_sigma**2)
        
        # trace_term = trace(A) * trace(G) / p_sigma^2
        # but trace(A) = sum(dA), trace(G) = sum(dG)
        trace_term = torch.trace(A) * torch.trace(G) / (p_sigma**2)
        
        # (q_weight_mu - p_weight_mu) is shaped (m*n,)
        delta_mu = q_weight_mu - p_mu
        # Reshape to (m, n) to match the Kronecker structure
        delta_mu_reshaped = delta_mu.view(m, n)
        
        # Quadratic term = trace(G_inv @ (delta_mu @ A_inv @ delta_mu^T))
        inside = delta_mu_reshaped @ A_inv @ delta_mu_reshaped.T
        quadratic_term = torch.trace(G_inv @ inside)
        
        # ----------------------------------------------------------------
        # 4) Put it all together
        # ----------------------------------------------------------------
        # This matches the original formula:
        # 0.5 * ((log_det_A + n*log_det_G - log_det_prior - m*n) 
        #        + trace_term + quadratic_term)
        kl = 0.5 * (
            (m*log_det_A + n * log_det_G - log_det_prior - m * n)
            + trace_term
            + quadratic_term
        )
        if kl < 0:
            kl = 0
            if flag == 'eval': kl = self.prev_kl
            self.training = False
            #print("FALSE")
        else:
            self.prev_kl = kl
            
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
        m = p_bias_mu.shape[0]

        log_det_q = torch.logdet(q_bias_cov)
        log_det_prior = m * torch.log(p_sigma**2) 
        trace_term = torch.trace(q_bias_cov) / p_sigma**2

        delta_mu = q_bias_mu - p_bias_mu
        quadratic_term = delta_mu.T @ torch.linalg.inv(q_bias_cov) @ delta_mu / p_sigma**2

        kl = 0.5 * (
            log_det_q - log_det_prior - m + trace_term + quadratic_term
        )

        return kl
    
    def _noisy_inversion(self, p_log_sigma, batch_size, damping=1e-2):
        """
        Computes the inverses of A and G.

        params:
            - p_log_sigma: log prior standard deviation
        """
        nu = torch.exp(2 * p_log_sigma)
        lam_scaled = torch.clamp(torch.tensor(math.sqrt(self.lam / (batch_size * nu))), min=1e-2)
        damp2 = (damping + lam_scaled) * torch.eye(self._G.size(0))
        damp1 = (damping + lam_scaled) * torch.eye(self._A.size(0))
        A, G = self._A + damp1, self._G + damp2
        

        # TODO: ignoring pi_l
        G_inv = torch.linalg.inv(G).double()
        A_inv = torch.linalg.inv(A).double()
        
        return A_inv, G_inv

