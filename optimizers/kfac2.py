import torch
import torch.optim as optim
from models.bayes_linear import BayesianLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class KFACOptimizer(optim.Optimizer):
    def __init__(self, 
            model,
            lr=0.01, 
            damping=1e-2, 
            beta2: float = 0.9,
            beta1: float = 0.9,
            gamma: float = 1e-4,
        ):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, 
                        damping=damping, 
                        beta1=beta1,
                        beta2=beta2, 
                        gamma=gamma)
        super().__init__(params, defaults)
        self.model = model

        # for each layer
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for layer in self.model.modules():
            if isinstance(layer, BayesianLinear):
                A_inv, G_inv = self._compute_layer_inv(layer)
                assert(A_inv is not None and G_inv is not None)

                for name, param in layer.named_parameters():
                    grad = param.grad
                    
                    if grad is None:
                        continue

                    if (layer.id, name) not in self.state:
                        self.state[(layer.id, name)] = grad
                        param.data -= self.defaults['lr'] * grad
                        continue

                    if name == 'q_bias_mu' or name == 'q_bias_log_sigma':
                        param.data -= self.defaults['lr'] * grad
                        continue


                    old_grad = self.state[(layer.id, name)]

                    # momentum-like updates of gradient with weight decay
                    prod = G_inv @ grad @ A_inv
                    grad_cur = self.defaults['beta2'] * old_grad + prod + self.defaults['gamma'] * param
                    self.state[(layer.id, name)] = grad_cur

                    param.data -= self.defaults['lr'] * grad_cur
        return loss
    
    def _compute_layer_inv(self, layer):
        """
        Updates layer state with new values of A, G. Returns A_inv and G_inv
        whose product is the kronecker approximation of the inverse hessian.
        """
        state = self.state
        # cnt indexes the layer
        A_old = state.get((layer.id,"A"))
        G_old = state.get((layer.id,"G"))
        A, G = layer._A, layer._G

        if A_old is not None and G_old is not None:
            A_new = (1 - self.defaults['beta1']) * A_old + self.defaults['beta1'] * A
            G_new = (1 - self.defaults['beta1']) * G_old + self.defaults['beta1'] * G

            # store updated per layer factors
            state[(layer.id,"A")] = A_new
            state[(layer.id,"G")] = G_new
            
            A_inv = self._invert_via_eig(A_new, self.defaults['damping'])
            G_inv = self._invert_via_eig(G_new, self.defaults['damping'])

        else:
            state[(layer.id,"A")] = A
            state[(layer.id,"G")] = G

            A_inv = self._invert_via_eig(A, self.defaults['damping'])
            G_inv = self._invert_via_eig(G, self.defaults['damping'])

        return A_inv, G_inv
        
    def _kron_mv(self, A, G, v):
        """
        Computes (A \oplus G) v where \oplus is the kronecker product. 
        It's more efficient than first computing the kronecker product
        """
        m, n = A.shape[0], G.shape[0]
        v = v.view(n, m)
        return (G @ v) @ A.t()
    
    def _invert_via_eig(self, M, damping=1e-4, eps=1e-10):
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

        #M_damped = M + damping * torch.eye(N, device=device, dtype=M.dtype)
        # Eigen-decomposition         
        d, Q = torch.linalg.eigh(M + damping * torch.eye(N, device=device, dtype=M.dtype), UPLO='U')
        d_clamped = torch.clamp(d, min=eps)
        inv_d = 1.0 / d_clamped
        
        # Reconstruct the inverse
        #   M^{-1} = Q diag(1/d_clamped) Q^T
        M_inv = Q @ torch.diag(inv_d) @ Q.t()
        return M_inv

   

