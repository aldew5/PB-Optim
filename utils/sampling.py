import torch


def sample_from_matrix_normal(q_mu, A, G, num_samples=1):
    """
    Sample from N(q_mu, A^{-1} ⊗ G^{-1}) using eigendecomposition to compute inverses.

    Parameters:
    - q_mu: torch.Tensor, shape (n, p), the mean matrix.
    - A: torch.Tensor, shape (n, n), the row covariance matrix (A, not its inverse).
    - G: torch.Tensor, shape (p, p), the column covariance matrix (G, not its inverse).
    - num_samples: int, the number of samples to generate (default: 1).

    Returns:
    - samples: torch.Tensor, shape (num_samples, n, p), the sampled matrices.
    """
    n, p = q_mu.shape

    # Eigendecomposition of A and G
    eigvals_A, eigvecs_A = torch.linalg.eigh(A)
    eigvals_G, eigvecs_G = torch.linalg.eigh(G)

    # Compute A^{-1/2} and G^{-1/2}
    A_inv_sqrt = eigvecs_A @ torch.diag(1.0 / torch.sqrt(eigvals_A)) @ eigvecs_A.T
    G_inv_sqrt = eigvecs_G @ torch.diag(1.0 / torch.sqrt(eigvals_G)) @ eigvecs_G.T

    # Generate standard normal samples
    z = torch.randn(num_samples, n, p)  # Shape: (num_samples, n, p)

    # Apply row covariance structure using A_inv_sqrt
    z = torch.einsum('ij,bjk->bik', A_inv_sqrt, z)  # Shape: (num_samples, n, p)

    # Apply column covariance structure using G_inv_sqrt
    samples = torch.einsum('bij,jk->bik', z, G_inv_sqrt.T)  # Shape: (num_samples, n, p)

    # Add the mean q_mu
    samples += q_mu

    return samples


def sample_from_kron_dist(q_mu, A, G, epsilon=1e-2):
    """
    Sample from a multivariate normal distribution with covariance A^{-1} ⊗ G^{-1}.
    
    Args:
        q_mu (torch.Tensor): Mean vector of size (m * n,).
        A (torch.Tensor): Covariance matrix A (n x n).
        G (torch.Tensor): Covariance matrix G (m x m).
        epsilon (float): Regularization constant to ensure positive-definiteness.
    
    Returns:
        torch.Tensor: Sample of size (m * n,).
    """
    if q_mu.ndim > 1:
        q_mu = q_mu.view(-1)
    
    n = A.shape[0]
    m = G.shape[0]
    
    with torch.no_grad():
        # Regularize A and G
        A = A + epsilon * torch.eye(n, device=A.device)
        G = G + epsilon * torch.eye(m, device=G.device)
        
        # Eigen decomposition and inverse square root
        dA, QA = torch.linalg.eigh(A)
        dG, QG = torch.linalg.eigh(G)
        
        dA_inv = 1.0 / torch.clamp(dA, min=epsilon)
        dG_inv = 1.0 / torch.clamp(dG, min=epsilon)
        
        sqrt_A_inv = QA @ torch.diag(torch.sqrt(dA_inv)) @ QA.T
        sqrt_G_inv = QG @ torch.diag(torch.sqrt(dG_inv)) @ QG.T
        
        # Generate samples
        Z = torch.randn(n, m, device=A.device)
        Y = sqrt_A_inv @ Z @ sqrt_G_inv.T
        samples = Y.flatten() + q_mu
        
        # Ensure sample dimensions match
        if samples.numel() != q_mu.numel():
            raise ValueError(
                f"Dimension mismatch: samples ({samples.numel()}) vs q_mu ({q_mu.numel()})"
            )
    
    return samples




def sample_from_kron_dist_fast(q_mu, A, G, epsilon=10):
    """
    Sample from a multivariate normal distribution with covariance A^{-1} ⊗ G^{-1},
    where we interpret the final (m*n)-dim vector as an (m x n) matrix.

    Args:
        q_mu (torch.Tensor): Mean vector of size (m * n,).
        A (torch.Tensor): Covariance matrix A, size (n x n).
        G (torch.Tensor): Covariance matrix G, size (m x m).
        epsilon (float): Regularization constant to ensure positive-definiteness.

    Returns:
        torch.Tensor: Sample of size (m * n,).
    """

    # ensure flattened mean
    if q_mu.ndim > 1:
        q_mu = q_mu.view(-1)

    m = G.shape[0] 
    n = A.shape[0]  

    # Regularize A and G for numerical stability
    A_reg = A + epsilon * torch.eye(n, device=A.device)
    G_reg = G + epsilon * torch.eye(m, device=G.device)

    # Eigendecomposition of A and G
    dA, QA = torch.linalg.eigh(A_reg)
    dG, QG = torch.linalg.eigh(G_reg)

    # Inverse square roots of eigenvalues
    dA_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(dA, min=epsilon))
    dG_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(dG, min=epsilon))

    # Draw a standard normal sample of shape (m x n) 
    # because we want the final shape to be (m x n).
    Z = torch.randn(m * n, device=A.device).view(m, n)

    Z_row_scaled = torch.diag(dG_inv_sqrt) @ Z        
    Z_left = QG @ Z_row_scaled                       
    Z_col_scaled = Z_left @ torch.diag(dA_inv_sqrt) 
    Z_transformed = Z_col_scaled @ QA.T               

    samples = Z_transformed.flatten() + q_mu

    # sanity
    if samples.numel() != q_mu.numel():
        raise ValueError(
            f"Dimension mismatch: samples ({samples.numel()}) vs q_mu ({q_mu.numel()})"
        )

    return samples



def sample_mvnd(mean, row_cov, col_cov):
    """
    Samples from a Matrix-Variate Normal Distribution using Cholesky decomposition.

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
    L_row = torch.linalg.cholesky(row_cov).double() # (m x m)
    L_col = torch.linalg.cholesky(col_cov).double() # (n x n)

    # Generate i.i.d. standard normal random matrix
    Z = torch.randn(m, n).double() # (m x n)

    # Transform the random matrix using the Kronecker structure
    sampled_matrix = mean + L_row @ Z @ L_col.T
    return sampled_matrix
