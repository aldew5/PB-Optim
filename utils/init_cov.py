import torch 

def init_cov(n, mean, std, epsilon=1e-5):
    """
    Initialize a symmetric covariance matrix with values from N(mean, std).

    Args:
        n (int): Size of the matrix (n x n).
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        epsilon (float): Small positive value added to the diagonal for positive definiteness.

    Returns:
        torch.Tensor: Symmetric covariance matrix (n x n).
    """
    # Step 1: Generate a random matrix from N(mean, std)
    random_matrix = torch.normal(mean, std, size=(n, n))
    
    # Step 2: Symmetrize the matrix
    symmetric_matrix = 0.5 * (random_matrix + random_matrix.T)
    
    # Step 3: Add epsilon to the diagonal for positive definiteness
    symmetric_matrix += epsilon * torch.eye(n)
    
    return symmetric_matrix