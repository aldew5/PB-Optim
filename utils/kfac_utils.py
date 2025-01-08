import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_layer_inv(layer, A=None, G=None, damping=1e-4):
        """
        Updates layer state with new values of A, G. Returns A_inv and G_inv
        whose product is the kronecker approximation of the inverse hessian.
        """
        if layer is not None:
              A, G = layer._A, layer._G
        #print(A, G)
        if A is not None and G is not None:
            A_root_inv, A_inv = compute_inv(A, damping)
            G_root_inv, G_inv = compute_inv(G, damping)
            return (A_root_inv, A_inv), (G_root_inv, G_inv)
        else:
            return None, None
        
def compute_inv(M, damping=1e-4, eps=1e-10):
        """
        Inverts a symmetric matrix M using its eigen-decomposition:
        M = Q diag(d) Q^T
        so:
        M^{-1} = Q diag(1/d) Q^T
        with small eigenvalues clamped to `eps` to avoid numerical issues.
        
        Args:
            M (torch.Tensor): Symmetric matrix (N x N).
            damping (float): Damping term added to the diagonal.
            eps (float): Minimum value for eigenvalues (for numerical stability).

        Returns:
            torch.Tensor: Inverse of (M + damping*I).
        """
        device = M.device
        N = M.size(0)

         # Ensure M is symmetric within a tolerance
        assert(torch.allclose(M, M.T, atol=1e-8))

        #M_damped = M + damping * torch.eye(N, device=device, dtype=M.dtype)
        # Eigen-decomposition         
        d, Q = torch.linalg.eigh(M + damping* torch.eye(N, device=device, dtype=M.dtype), UPLO='U')
        d_clamped = torch.clamp(d, min=eps)
        inv_d = 1.0 / d_clamped
        
        # Reconstruct the inverse
        #   M^{-1} = Q diag(1/d_clamped) Q^T
        #M_root_inv = Q @ torch.diag(torch.sqrt(inv_d)) @ Q.t()
        M_inv = Q @ torch.diag(inv_d) @ Q.t()

        return M_inv

import torch

def sample_from_kron_dist(q_weight_mu, A, G, epsilon=1e-4):
    """
    Sample from a multivariate normal distribution with covariance A ⊗ G
    without explicitly forming the Kronecker product.

    Args:
        q_weight_mu (torch.Tensor): Mean vector of size (m * n,).
        A (torch.Tensor): Covariance matrix A (n x n).
        G (torch.Tensor): Covariance matrix G (m x m).
        epsilon (float): Regularization constant to ensure positive-definiteness.

    Returns:
        torch.Tensor: Sample of size (m * n,).
    """
    # Ensure q_weight_mu is a 1D tensor
    if q_weight_mu.ndim > 1:
        q_weight_mu = q_weight_mu.view(-1)

    n = A.shape[0]
    m = G.shape[0]

    # Regularize A and G to ensure positive definiteness
    A = A + epsilon * torch.eye(n, device=A.device, dtype=A.dtype)
    G = G + epsilon * torch.eye(m, device=G.device, dtype=G.dtype)

    # Compute Cholesky factors
    try:
        sqrt_A = torch.linalg.cholesky(A).double() # n x n
    except RuntimeError as e:
        raise RuntimeError(f"Cholesky decomposition failed for A: {e}")

    try:
        sqrt_G = torch.linalg.cholesky(G).double()  # m x m
    except RuntimeError as e:
        raise RuntimeError(f"Cholesky decomposition failed for G: {e}")

    # Generate standard normal samples
    Z = torch.randn(n, m, device=A.device, dtype=sqrt_A.dtype)  # n x m

    # Transform samples using Kronecker structure to achieve Cov(Y) = A ⊗ G
    Y = sqrt_A @ Z @ sqrt_G.T  # n x m
    samples = Y.flatten()  # Flatten to (n * m,)

    # Ensure samples have the same dtype as q_weight_mu
    #samples = samples.to(dtype=q_weight_mu.dtype)
    


    if samples.numel() != q_weight_mu.numel():
        raise ValueError(
            f"Dimension mismatch: samples ({samples.numel()}) vs q_weight_mu ({q_weight_mu.numel()})"
        )
    
    return samples + q_weight_mu



def sample_mvnd(mean, row_cov, col_cov):
    """
    Samples from a Matrix-Variate Normal Distribution.

    Args:
        mean (torch.Tensor): Mean matrix (m x n).
        row_cov (torch.Tensor): Row covariance matrix (m x m).
        col_cov (torch.Tensor): Column covariance matrix (n x n).
    
    Returns:
        torch.Tensor: Sampled matrix (m x n).
    """
    # Shape of the mean matrix
    m, n = mean.shape

    # Cholesky decomposition or square root of covariance matrices
    L_row = torch.linalg.cholesky(row_cov) # (m x m)
    L_col = torch.linalg.cholesky(col_cov) # (n x n)

    # Generate i.i.d. standard normal random matrix
    Z = torch.randn(m, n) # (m x n)

    # Transform the random matrix using the Kronecker structure
    sampled_matrix = mean + L_row @ Z @ L_col.T
    return sampled_matrix

"""

def kl_divergence_kfac(mu_p, A_p, G_p, mu_q, A_q, G_q, epsilon=1e-5):
   
    A_q = A_q + epsilon * torch.eye(A_q.shape[0], device=A_q.device)
    G_q = G_q + epsilon * torch.eye(G_q.shape[0], device=G_q.device)

    # Dimensions
    n = A_p.shape[0]  # Input features
    m = G_p.shape[0]  # Output features
    
    # Reshape mean difference
    delta_mu = mu_q - mu_p
    delta_mu_reshaped = delta_mu.view(m, n)  # Reshape to (m, n)
    
    # Log determinants
    log_det_A_p = torch.logdet(A_p)
    log_det_G_p = torch.logdet(G_p)
    log_det_A_q = torch.logdet(A_q)
    log_det_G_q = torch.logdet(G_q)
    
    # Trace terms
    trace_A = torch.trace(torch.linalg.inv(A_q) @ A_p)
    trace_G = torch.trace(torch.linalg.inv(G_q) @ G_p)
    
    # Quadratic term
    inv_G_q = torch.linalg.inv(G_q)
    inv_A_q = torch.linalg.inv(A_q)
    quadratic_term = torch.trace(inv_G_q @ delta_mu_reshaped @ inv_A_q @ delta_mu_reshaped.T)
    
    # KL divergence
    kl = 0.5 * (
        (log_det_A_q + n * log_det_G_q) - (log_det_A_p + n * log_det_G_p) - m * n
        + trace_A * trace_G
        + quadratic_term
    )
    
    return kl.item()
"""