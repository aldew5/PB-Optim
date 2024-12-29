import torch
import torch.optim as optim
from models.bayes_linear import BayesianLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class KFACOptimizer(optim.Optimizer):
    def __init__(self, 
            model,
            lr=0.01, 
            damping=1e-1, 
            beta1: float = 0.9,
            beta2: float = 0.99999,
            weight_decay: float = 1e-4,
            ess: int = 1e6
        ):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, 
                        damping=damping, 
                        beta1=beta1,
                        beta2=beta2, 
                        weight_decay=weight_decay,
                        ess=ess)
        super().__init__(params, defaults)
        self.model = model
        self._numel, self._device, self._dtype = self._get_param_configs()

        # for each layer
        self.state = {}
        self._init_buffers()

    def _init_buffers(self):
        for group in self.param_groups:
            numel = group["numel"]

            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for layer in self.model.modules():
            if isinstance(layer, BayesianLinear):
                A_inv, G_inv = self._compute_layer_inv(layer)
                if (A_inv is None): 
                    continue
                
                for name, param in layer.named_parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    
                    # only condition weight posterior updates
                    if name[2] == 'w':
                        old_grad = self.state[(layer.id, name)] if (layer.id, name) in self.state else torch.zeros_like(grad)
                        grad_cur = self.defaults['beta1'] * old_grad + (1-self.defaults['beta1']) * grad
                        self.state[(layer.id, name)] = grad_cur
                        #print("TOP", layer.state[name].shape)
                        natural_grad = self._kron_mv(A_inv, G_inv, grad_cur.view(-1))

                        param.data -= self.defaults['lr'] * (natural_grad + self.defaults['weight_decay'] * param.data)
                       

                    elif name[2] == 'b': # biases
                        old_grad =  self.state[(layer.id, name)] if (layer.id, name) in self.state else torch.zeros_like(grad)
                        #print("HERE", old_grad.shape, grad.shape)
                        grad_cur = self.defaults['beta1'] * old_grad + (1-self.defaults['beta1']) * grad
                        self.state[(layer.id, name)] = grad_cur
                        #print("BOTTOM", layer.state[name].shape)

                        param.data -= self.defaults['lr'] * (grad_cur + self.defaults['weight_decay'] * param.data)
                    else:
                        raise ValueError(f"Unknown parameter name: {name}")
                    
        #old_sigma = self.state['p_log_sigma'] if 'p_log_sigma' in state else torch.zeros_like(self.p_lo)
        if 'momentum' not in self.state:
            self.state['momentum'] = torch.zeros_like(self.model.p_log_sigma.grad)

        old_grad = self.state['momentum']
        grad = self.model.p_log_sigma.grad
        grad_cur = self.defaults['beta1'] * old_grad + grad * (1 - self.defaults['beta1']) 
        self.state['momentum'] = grad_cur    
    
        self.model.p_log_sigma.data -= self.defaults['lr'] * (grad_cur + self.defaults['weight_decay'] * self.model.p_log_sigma.data)

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
            A_inv = torch.inverse(A_new + self.defaults['damping'] * torch.eye(A_new.size(0)).to(A_new.device))
            G_inv = torch.inverse(G_new + self.defaults['damping'] * torch.eye(G_new.size(0)).to(G_new.device))
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
    
    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(
                "Parameters are on different devices: "
                f"{[str(d) for d in devices]}"
            )
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(
                "Parameters are on different dtypes: "
                f"{[str(d) for d in dtypes]}"
            )
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype