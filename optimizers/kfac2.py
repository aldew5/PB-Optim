import torch
import torch.optim as optim
from models.bayes_linear import BayesianLinear
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class KFACOptimizer(optim.Optimizer):
    """
    Using block diagonal kfactored approximation of the Fisher

    lam: damping used in inversion of kfactors
    """
    def __init__(self, 
            model,
            lr=0.01, 
            lam=1e-2, 
            beta2: float=0.9,
            beta1: float=0.9,
            weight_decay=1e-4
        ):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, 
                        lam=lam, 
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.model = model
        self.gamma = math.sqrt(lam + lr)
        # count iterations
        self.k = 1

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

                    elif name == 'q_bias_mu' or name == 'q_bias_log_sigma':
                        param.data -= self.defaults['lr'] * grad
                        
                    else:
                        old_grad = self.state[(layer.id, name)]

                        # momentum-like updates of gradient with weight decay
                        prod = G_inv @ grad @ A_inv
                        grad_cur = self.defaults['beta2'] * old_grad + prod + self.defaults['weight_decay'] * param
                        self.state[(layer.id, name)] = grad_cur

                        param.data -= self.defaults['lr'] * grad_cur
        return loss
    
    def _compute_layer_inv(self, layer):
        """
        Updates layer state with new values of A, G. Returns A_inv and G_inv
        whose product is the kronecker approximation of the inverse hessian.
        """
        # retrieve kronecker approximations for the current layer
        # NOTE: we are using block approximations A_ii, G_ii for layer i
        A_old = self.state.get((layer.id,"A"))
        G_old = self.state.get((layer.id,"G"))
        A, G = layer._A, layer._G

        if A_old is not None and G_old is not None:
            A_new = (1 - self.defaults['beta1']) * A_old + self.defaults['beta1'] * A
            G_new = (1 - self.defaults['beta1']) * G_old + self.defaults['beta1'] * G

            # store updated per layer factors
            self.state[(layer.id,"A")] = A_new
            self.state[(layer.id,"G")] = G_new

            # see grosse et al for calculation
            pi = torch.sqrt((torch.trace(A_new)/(A_new.size(0) + 1))/(torch.trace(G_new)/(G_new.size(0))))
            A_new_damped = A_new + pi * self.gamma * torch.eye(A_new.size(0), device=A_new.device)
            G_new_damped = G_new + 1.0/pi * self.gamma * torch.eye(G_new.size(0), device=G_new.device)

            
            A_inv = self._invert_via_eig(A_new_damped)
            G_inv = self._invert_via_eig(G_new_damped)

        # first iteration
        else:
            self.state[(layer.id,"A")] = A
            self.state[(layer.id,"G")] = G

            pi = torch.sqrt((torch.trace(A)/(A.size(0) + 1))/(torch.trace(G)/(G.size(0))))
            A_damped = A + pi * self.gamma * torch.eye(A.size(0), device=A.device)
            G_damped = G + 1.0/pi * self.gamma * torch.eye(G.size(0), device=G.device)

            A_inv = self._invert_via_eig(A_damped)
            G_inv = self._invert_via_eig(G_damped)

        return A_inv, G_inv
        
    
    def _invert_via_eig(self, M, eps=1e-8):
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

        # Eigen-decomposition         
        d, Q = torch.linalg.eigh(M, UPLO='U')
        d_clamped = torch.clamp(d, min=eps)
        inv_d = 1.0 / d_clamped
        
        # Reconstruct the inverse
        #   M^{-1} = Q diag(1/d_clamped) Q^T
        M_inv = Q @ torch.diag(inv_d) @ Q.t()
        return M_inv

   

