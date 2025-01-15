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

def sample_from_kron_dist(q_mu, A, G, epsilon=1e-2):
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
    #print("Q MU", q_mu)
    # Ensure q_weight_mu is a 1D tensor
    if q_mu.ndim > 1:
        q_mu = q_mu.view(-1)

    n = A.shape[0]
    m = G.shape[0]


    # Regularize A and G to ensure positive definiteness
    A = A + max(A.max() * epsilon, epsilon) * torch.eye(n, device=A.device, dtype=A.dtype)
    G = G + max(G.max() * epsilon, epsilon) * torch.eye(m, device=G.device, dtype=G.dtype)
    #A = A / A.norm()
    #G = G / G.norm()        
    #print(A, G, torch.linalg.eigvals(A))

    # Compute Cholesky factors
    try:
        sqrt_A = torch.linalg.cholesky(A) # n x n
    except RuntimeError as e:
        raise RuntimeError(f"Cholesky decomposition failed for A: {e}")

    try:
        sqrt_G = torch.linalg.cholesky(G)  # m x m
    except RuntimeError as e:
        raise RuntimeError(f"Cholesky decomposition failed for G: {e}")
    
    
    # Generate standard normal samples
    Z = torch.randn(n, m, device=A.device, dtype=sqrt_A.dtype)  # n x m

    # Transform samples using Kronecker structure to achieve Cov(Y) = A ⊗ G
    #print("IN", sqrt_A, sqrt_G, Z)
    Y = sqrt_A @ Z @ sqrt_G.T  # n x m
    samples = Y.flatten()  # Flatten to (n * m,)
   # print("samples", samples, q_mu)

    # Ensure samples have the same dtype as q_weight_mu
    #samples = samples.to(dtype=q_weight_mu.dtype)

    if samples.numel() != q_mu.numel():
        raise ValueError(
            f"Dimension mismatch: samples ({samples.numel()}) vs q_weight_mu ({q_mu.numel()})"
        )
    
    return samples + q_mu



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


import torch
import torch.nn as nn
import torch.nn.functional as F


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





