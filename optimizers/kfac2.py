import torch
import torch.optim as optim
from models.bayes_linear import BayesianLinear
from utils.kfac_utils import *
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class KFACOptimizer(optim.Optimizer):
    """
    params:
        - lam is the kl weighting.
        - 
    """
    def __init__(self, 
            model,
            lr=0.01, 
            damping=1e-1, 
            beta1: float = 0.9,
            beta2: float = 0.99999,
            weight_decay: float = 1e-4,
            lam: int = 1e-1,
            batch_size=100,
            t_stat=10,
            t_inv=10,
            gamma_ex=1e-4
        ):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, 
                        damping=damping, 
                        beta1=beta1,
                        beta2=beta2, 
                        weight_decay=weight_decay,
                        lam=lam, 
                        batch_size=batch_size,
                        t_stat=t_stat,
                        t_inv=t_inv,
                        gamma_ex=gamma_ex)
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
                A_inv, G_inv = self._noisy_inversion(layer)
                
                for name, param in layer.named_parameters():
                    grad = param.grad
                    
                    if grad is None:
                        continue
                    
                    # only condition weight posterior updates
                    if name[2] == 'w':
                        layer._A = (1 - self.defaults['beta1']) * layer._A + self.defaults['beta1']

                        """
                        old_grad = self.state[(layer.id, name)] if (layer.id, name) in self.state else torch.zeros_like(grad)
                        grad_cur = self.defaults['beta1'] * old_grad + (1-self.defaults['beta1']) * grad
                        self.state[(layer.id, name)] = grad_cur
                        natural_grad = self._kron_mv(A_inv, G_inv, grad_cur.view(-1))

                        param.data -= self.defaults['lr'] * natural_grad #+ self.defaults['weight_decay'] * param.data)
                        """
                       

                    elif name[2] == 'b': # biases
                        old_grad =  self.state[(layer.id, name)] if (layer.id, name) in self.state else torch.zeros_like(grad)
                        grad_cur = self.defaults['beta1'] * old_grad + (1-self.defaults['beta1']) * grad
                        self.state[(layer.id, name)] = grad_cur

                        param.data -= self.defaults['lr'] * grad_cur #+ self.defaults['weight_decay'] * param.data)
                    else:
                        raise ValueError(f"Unknown parameter name: {name}")
                    
        #old_sigma = self.state['p_log_sigma'] if 'p_log_sigma' in state else torch.zeros_like(self.p_lo)
        if 'momentum' not in self.state:
            self.state['momentum'] = torch.zeros_like(self.model.p_log_sigma.grad)

        old_grad = self.state['momentum']
        grad = self.model.p_log_sigma.grad
        grad_cur = self.defaults['beta1'] * old_grad + grad * (1 - self.defaults['beta1']) 
        self.state['momentum'] = grad_cur    
    
        self.model.p_log_sigma.data -= self.defaults['lr'] * grad_cur #+ self.defaults['weight_decay'] * self.model.p_log_sigma.data)

        return loss
    
    

    
    def _smooth_update(self, old, new, decay):
        """Apply an exponential moving average to smooth updates."""
        if old is None:
            return new
        return decay * old + (1 - decay) * new
    
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

        #print(A, G)
        if A is not None and G is not None:
            A_new = self._smooth_update(
                A_old.detach() if A_old is not None else None,
                A.detach(),
                self.defaults['beta2']
                )
            G_new = self._smooth_update(
                G_old.detach() if G_old is not None else None,
                G.detach(),
                self.defaults['beta2']
                )

            state[(layer.id,"A")] = A_new
            state[(layer.id,"G")] = G_new
            #print(A_new, G_new)
            A_inv = self._invert_via_eig(A_new, self.defaults['damping'])
            G_inv = self._invert_via_eig(G_new, self.defaults['damping'])
            return A_inv, G_inv
        else:
            return None, None
        
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
        d, Q = torch.linalg.eigh(M + damping* torch.eye(N, device=device, dtype=M.dtype), UPLO='U')
        d_clamped = torch.clamp(d, min=eps)
        inv_d = 1.0 / d_clamped
        
        # Reconstruct the inverse
        #   M^{-1} = Q diag(1/d_clamped) Q^T
        M_inv = Q @ torch.diag(inv_d) @ Q.t()
        return M_inv

   

