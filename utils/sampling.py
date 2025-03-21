import torch


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


def sample_matrix_normal(q_mu, A_inv, G_inv, lam, N, epsilon=1e-8, precision="float32"):
    """
    Sample from a multivariate normal distribution
        N(q_mu, G^{-1} ⊗ A^{-1})
    which is equivalent to the matrix normal
        MN(q_mu, A^{-1}, G^{-1}),
    where we interpret the final (m*n)-dim vector as an (m x n) matrix.
    
    Here A is (n x n) and G is (m x m). The interpretation is that the
    column covariance is A^{-1} and the row covariance is G^{-1}, so that
    vec(X) ~ N(vec(q_mu), G^{-1} ⊗ A^{-1}).

    Args:
        q_mu (torch.Tensor): Mean vector of size (m * n,).
        A (torch.Tensor): A matrix of size (n x n). Its inverse gives the column covariance.
        G (torch.Tensor): A matrix of size (m x m). Its inverse gives the row covariance.
        epsilon (float): Regularization constant to ensure positive-definiteness.

    Returns:
        torch.Tensor: Sample vector of size (m * n,).
    """
    # vec(q_mu)
    q_mu = q_mu.view(-1)
    m = G_inv.shape[0] 
    n = A_inv.shape[0] 

    # in the first run the optimizer won't yet have calculated 
    # sqrt_G_inv
    
    # Compute eigen-decomposition of A_inv and G_inv.
    d_A, Q_A = torch.linalg.eigh(A_inv, UPLO='U')
    d_G, Q_G = torch.linalg.eigh(G_inv, UPLO='U')
    
    # Clamp eigenvalues for numerical stability.
    d_A = torch.clamp(d_A, min=epsilon)
    d_G = torch.clamp(d_G, min=epsilon)
    
    # Compute the square-root factors:
    # sqrt_A_inv such that sqrt_A_inv @ sqrt_A_inv^T = A_inv.
    sqrt_A_inv = Q_A @ torch.diag(d_A.sign() * torch.sqrt(d_A.abs())) @ Q_A.t()
    sqrt_G_inv = Q_G @ torch.diag(d_G.sign() * torch.sqrt(d_G.abs())) @ Q_G.t()
    
    # Draw a standard normal sample Z of shape (m, n).
    Z = torch.randn(m, n, device=A_inv.device, dtype=getattr(torch, precision))
    sqrt_A_inv, sqrt_G_inv = sqrt_A_inv.type(getattr(torch, precision)), sqrt_G_inv.type(getattr(torch, precision))
    
    # Sample from the matrix normal:
    #   X = sqrt_G_inv @ Z @ sqrt_A_inv^T
    X =  lam/N * sqrt_G_inv @ Z @ sqrt_A_inv.t()
    
    # Flatten X and add the mean.
    sample_vec = X.flatten() + q_mu
    
    if sample_vec.numel() != q_mu.numel():
        raise ValueError(
            f"Dimension mismatch: sample has {sample_vec.numel()} elements, "
            f"but q_mu has {q_mu.numel()} elements."
        )
    
    return sample_vec


def sample_from_kron_dist_fast(q_mu, A, G, epsilon=1e-2):
    """
    Sample from a multivariate normal distribution N(q_mu, A^{-1} ⊗ G^{-1}) = MN(q_mu, G^{-1}, A^{-1}),
    where we interpret the final (m*n)-dim vector as an (m x n) matrix.

    Args:
        q_mu (torch.Tensor): Mean vector of size (m * n,).
        A (torch.Tensor): Covariance matrix A, size (n x n).
        G (torch.Tensor): Covariance matrix G, size (m x m).
        epsilon (float): Regularization constant to ensure positive-definiteness.

    Returns:
        torch.Tensor: Sample of size (m * n,).
    
    1. double-check KL with damping 
    2. Sampling at mean plus empirical risk with normal sample
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



def sample_activations_kron_fast(
    q_mu: torch.Tensor,  # Mean of vec(W) in shape (m*n,) or (m, n)
    X: torch.Tensor,     # "Input" matrix of shape (M, m)
    A: torch.Tensor,     # Kronecker factor for columns (n x n)
    G: torch.Tensor,     # Kronecker factor for rows (m x m)
    epsilon: float = 1e-1
) -> torch.Tensor:
    """
    Sample B = X * W where W ~ N(q_mu, (A^-1 ⊗ G^-1)), i.e. Kronecker-factorized
    Gaussian for W in R^{m x n}, without explicitly forming W.
    
    Args:
        q_mu   : Flattened mean for W (m*n,) or (m,n).
        X      : (M x m) matrix we multiply by W.
        A      : (n x n) factor for columns of W.
        G      : (m x m) factor for rows of W.
        epsilon: Regularization constant.

    Returns:
        B_samp : (M x n) sample drawn from the distribution of B.
    """
    
    m = G.shape[0]
    n = A.shape[0]
    #print(m, n, q_mu.size())

    # -- 1. Compute B's mean directly: B_mean = X @ W_mu
    B_mean = X @ q_mu.T  # shape (M x n)

    # -- 2. We need an eigen-factorization of G and A (regularized).
    A_reg = A + epsilon * torch.eye(n, device=A.device, dtype=A.dtype)
    G_reg = G + epsilon * torch.eye(m, device=G.device, dtype=G.dtype)

    dA, QA = torch.linalg.eigh(A_reg) 
    dG, QG = torch.linalg.eigh(G_reg)  

    # -- 3. Inverse sqrt of eigenvalues
    dA_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(dA, min=epsilon))
    dG_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(dG, min=epsilon))

    # -- 4. Draw standard normal Z of shape (m x n) 
    Z = torch.randn(m, n, device=q_mu.device, dtype=q_mu.dtype)

    # -- 5. Apply the Kronecker-factor “square root” transform for W:
    #        W_samp = QG diag(dG_inv_sqrt) * Z * diag(dA_inv_sqrt) QA^T
    #   This would be a sample from N(0, A^-1 ⊗ G^-1) if we omit the mean.
    Z_row_scaled = torch.diag(dG_inv_sqrt) @ Z         
    Z_left       = QG @ Z_row_scaled                     
    Z_col_scaled = Z_left @ torch.diag(dA_inv_sqrt)      
    W_noise      = Z_col_scaled @ QA.T                    

    B_noise = X @ W_noise.T 
    B_samp = B_mean + B_noise  

    return B_samp


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
