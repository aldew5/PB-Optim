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

    # Flatten mean if necessary
    if q_mu.ndim > 1:
        q_mu = q_mu.view(-1)

    # Sizes
    m = G.shape[0]  # row dimension
    n = A.shape[0]  # column dimension

    # Regularize A and G for numerical stability
    A_reg = A + epsilon * torch.eye(n, device=A.device)
    G_reg = G + epsilon * torch.eye(m, device=G.device)
    #print(A_reg.diagonal(), G_reg.diagonal())

    # Eigendecomposition of A and G
    dA, QA = torch.linalg.eigh(A_reg)
    dG, QG = torch.linalg.eigh(G_reg)

    # Inverse square roots of eigenvalues
    dA_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(dA, min=epsilon))
    dG_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(dG, min=epsilon))

    # Draw a standard normal sample of shape (m x n) 
    # because we want the final shape to be (m x n).
    Z = torch.randn(m * n, device=A.device).view(m, n)

    # Build the transformation so that Cov(vec(Z_transformed)) = A^-1 ⊗ G^-1.
    #
    # Recall that for an (m x n) matrix M:
    #   vec(QG * M * QA^T) = (QA ⊗ QG) * vec(M),
    # and each QG, QA is orthogonal, so the eigen decomposition
    #   G = QG diag(dG) QG^T => G^-1 = QG diag(1/dG) QG^T
    #   A = QA diag(dA) QA^T => A^-1 = QA diag(1/dA) QA^T.
    #
    # To get G^-1 for rows and A^-1 for columns, we do:
    #   Z_transformed = QG (diag(dG_inv_sqrt) Z diag(dA_inv_sqrt)) QA^T
    #
    # But for simplicity, we often multiply each piece step by step:
    #
    #   diag(dG_inv_sqrt) @ Z => multiply each row by sqrt of 1/dG (since Z is (m x n))
    #   QG @ (...) => rotate by QG
    #   (... ) @ diag(dA_inv_sqrt) => multiply each column by sqrt of 1/dA
    #   (... ) @ QA^T => rotate by QA
    #
    # The net result is that vec(Z_transformed) has covariance A^-1 ⊗ G^-1.
    #
    # A quick route is:  Z_transformed = QG * (diag(dG_inv_sqrt) * Z * diag(dA_inv_sqrt)) * QA^T

    Z_row_scaled = torch.diag(dG_inv_sqrt) @ Z        # shape (m x n)
    Z_left = QG @ Z_row_scaled                        # shape (m x n)
    Z_col_scaled = Z_left @ torch.diag(dA_inv_sqrt)   # shape (m x n)
    Z_transformed = Z_col_scaled @ QA.T               # shape (m x n)

    # Flatten and add mean
    samples = Z_transformed.flatten() + q_mu

    # Safety check
    if samples.numel() != q_mu.numel():
        raise ValueError(
            f"Dimension mismatch: samples ({samples.numel()}) vs q_mu ({q_mu.numel()})"
        )

    #print(samples.diagonal())
    return samples





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
    L_row = torch.linalg.cholesky(row_cov).double() # (m x m)
    L_col = torch.linalg.cholesky(col_cov).double() # (n x n)

    # Generate i.i.d. standard normal random matrix
    Z = torch.randn(m, n).double() # (m x n)

    # Transform the random matrix using the Kronecker structure
    sampled_matrix = mean + L_row @ Z @ L_col.T
    return sampled_matrix


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_stat(aa, m_aa, stat_decay):
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)
  

class ComputeMatGrad:

    @classmethod
    def __call__(cls, input, grad_output, layer):
        grad = cls.linear(input, grad_output, layer)
        #elif isinstance(layer, nn.Conv2d):
        #    grad = cls.conv2d(input, grad_output, layer)
        #else:
        #    raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with q_bias_mu])
        """
        with torch.no_grad():
            if layer.q_bias_mu is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.unsqueeze(1)
            grad_output = grad_output.unsqueeze(2)
            grad = torch.bmm(grad_output, input)
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with q_bias_mu])
        :return:
        """
        with torch.no_grad():
            input = _extract_patches(input, layer.kernel_size, layer.stride, layer.padding)
            input = input.view(-1, input.size(-1))  # b * hw * in_c*kh*kw
            grad_output = grad_output.transpose(1, 2).transpose(2, 3)
            grad_output = try_contiguous(grad_output).view(grad_output.size(0), -1, grad_output.size(-1))
            # b * hw * out_c
            if layer.q_bias_mu is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.view(grad_output.size(0), -1, input.size(-1))  # b * hw * in_c*kh*kw
            grad = torch.einsum('abm,abn->amn', (grad_output, input))
        return grad


class ComputeCovA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        #if isinstance(layer, BayesianLinear):
        cov_a = cls.linear(a, layer)
        #elif isinstance(layer, nn.Conv2d):
        #    cov_a = cls.conv2d(a, layer)
        #else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
        #    cov_a = None

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.q_bias_mu is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        #if layer.q_bias_mu is not None:
        a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)


class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        #if isinstance(layer, nn.Conv2d):
        #cov_g = cls.conv2d(g, layer, batch_averaged)
        #elif isinstance(layer, BayesianLinear):
        cov_g = cls.linear(g, layer, batch_averaged)
        #else:
        #    cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g



if __name__ == '__main__':
    def test_ComputeCovA():
        pass

    def test_ComputeCovG():
        pass





