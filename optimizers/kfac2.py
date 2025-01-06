import torch
import torch.optim as optim
from utils.kfac_utils import *
import math

class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        # Specify which modules to apply K-FAC to
        self.known_modules = {'BayesianLinear'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        # Separate state dictionaries for covariance and eigen decompositions
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

        # Initialize separate layer state
        self.layer_state = {}

    def _prepare_model(self):
        """
        Register hooks on BayesianLinear layers to capture activation
        and gradient info for K-FAC.
        """
        count = 0
        print("=> KFAC on BayesianLinear layers only. Found:")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                # forward_pre_hook for input
                module.register_forward_pre_hook(self._save_input)
                # backward hook for gradient output
                module.register_backward_hook(self._save_grad_output)
                print(f"({count}) {module}")
                count += 1

    def _save_input(self, module, input):
        """
        Capture the layer input to compute CovA. We'll do this every TCov steps.
        """
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            input_data = input[0].data
            if hasattr(module, 'q_bias_mu') and module.q_bias_mu is not None:
                # Augment input with a column of ones for the bias term
                ones = torch.ones((input_data.size(0), 1), device=input_data.device, dtype=input_data.dtype)
                input_data = torch.cat([input_data, ones], dim=1)
            try:
                aa = self.CovAHandler.compute_cov_a(input_data, module)
            except ValueError as e:
                print(f"[ERROR] Step {self.steps}, Module {module}, CovA computation failed: {e}")
                aa = torch.eye(input_data.size(1), device=input_data.device, dtype=input_data.dtype)
            
            print(f"[DEBUG] Step {self.steps}, Module {module}, CovA: {aa}")
            print("Input Data:", input_data)
            
            if aa is None:
                print(f"[WARNING] CovAHandler returned None for module {module}. Using identity matrix.")
                in_dim = module.in_features + 1 if hasattr(module, 'q_bias_mu') and module.q_bias_mu is not None else module.in_features
                aa = torch.eye(in_dim, device=input_data.device, dtype=input_data.dtype)
            
            if self.steps == 0:
                # Initialize m_aa with identity matrix of correct size
                in_dim = aa.size(0)
                self.m_aa[module] = torch.diag(torch.ones(in_dim, device=aa.device, dtype=aa.dtype))
            
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        """
        Capture the gradient w.r.t. layer outputs to compute CovG.
        We'll do this every TCov steps.
        """
        if self.steps % self.TCov == 0:
            try:
                gg = self.CovGHandler.compute_cov_g(grad_output[0].data, module, self.batch_averaged)
            except ValueError as e:
                print(f"[ERROR] Step {self.steps}, Module {module}, CovG computation failed: {e}")
                gg = torch.eye(module.out_features, device=grad_output[0].device, dtype=grad_output[0].dtype)
            
            print(f"[DEBUG] Step {self.steps}, Module {module}, CovG: {gg}")
            if gg is None:
                print(f"[WARNING] CovGHandler returned None for module {module}. Using identity matrix.")
                out_dim = module.out_features
                gg = torch.eye(out_dim, device=grad_output[0].device, dtype=grad_output[0].dtype)
            
            if self.steps == 0:
                # Initialize m_gg with identity matrix
                out_dim = gg.size(0)
                self.m_gg[module] = torch.diag(torch.ones(out_dim, device=gg.device, dtype=gg.dtype))
            
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _update_inv(self, m):
        """
        Perform eigen-decomposition on the CovA and CovG factors for module m.
        Store Q_a, d_a and Q_g, d_g for quick inversion.
        """
        eps = 1e-10  # for numerical stability
        try:
            # Use torch.linalg.eigh instead of torch.symeig
            self.d_a[m], self.Q_a[m] = torch.linalg.eigh(self.m_aa[m], UPLO='L')
            self.d_g[m], self.Q_g[m] = torch.linalg.eigh(self.m_gg[m], UPLO='L')
        except RuntimeError as e:
            print(f"[ERROR] Eigen-decomposition failed for module {m}: {e}")
            # Handle failure: increase damping, skip update, etc.
            return
        
        # Clamp small eigenvalues for numerical stability
        self.d_a[m].clamp_(min=eps)
        self.d_g[m].clamp_(min=eps)

    @staticmethod
    def _get_matrix_form_grad(layer):
        """
        Gather q_weight_mu.grad and q_bias_mu.grad into a 2D matrix.
        """
        # Ensure gradients are computed
        if layer.q_weight_mu.grad is None:
            raise ValueError(f"q_weight_mu.grad is None for layer {layer}")
        
        w_grad = layer.q_weight_mu.grad.data
        if layer.q_bias_mu is not None:
            if layer.q_bias_mu.grad is None:
                raise ValueError(f"q_bias_mu.grad is None for layer {layer}")
            b_grad = layer.q_bias_mu.grad.data.view(-1, 1)
            # Concatenate along columns to form [out_features, in_features + 1]
            p_grad_mat = torch.cat([w_grad, b_grad], dim=1)
        else:
            # No bias
            p_grad_mat = w_grad
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        Compute the natural gradient using the eigen-decomposed Fisher factors.
        """
        # Transform to eigen-basis
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        # Divide by (d_g * d_a + damping)
        denom = (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v2 = v1 / denom
        # Transform back to original space
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

        # Split into weight and bias gradients
        if m.q_bias_mu is not None:
            w = v[:, :-1].view(m.q_weight_mu.grad.data.size())
            b = v[:, -1:].view(m.q_bias_mu.grad.data.size())
            return [w, b]
        else:
            w = v.view(m.q_weight_mu.grad.data.size())
            return [w]

    def _kl_clip_and_update_grad(self, updates, lr):
        """
        Scale the natural gradients to ensure KL divergence constraint.
        """
        vg_sum = 0.0
        for layer in self.modules:
            v = updates[layer]
            # v[0] -> weight update
            vg_sum += (v[0] * layer.q_weight_mu.grad.data * lr ** 2).sum().item()
            if layer.q_bias_mu is not None and len(v) > 1:
                vg_sum += (v[1] * layer.q_bias_mu.grad.data * lr ** 2).sum().item()
        
        if vg_sum > 0:
            nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))
        else:
            nu = 1.0

        # Apply the scaling factor nu to the gradients
        for layer in self.modules:
            v = updates[layer]
            with torch.no_grad():
                layer.q_weight_mu.grad.data.copy_(v[0])
                layer.q_weight_mu.grad.data.mul_(nu)
                if layer.q_bias_mu is not None and len(v) > 1:
                    layer.q_bias_mu.grad.data.copy_(v[1])
                    layer.q_bias_mu.grad.data.mul_(nu)

    def _step_sgd(self, closure):
        """
        Perform a standard SGD step with momentum and weight decay.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                # Update parameter
                p.data.add_(-lr, d_p)

    def step(self, closure=None):
        """
        Perform one K-FAC update step:
        1) Update inverses periodically.
        2) Gather gradients in matrix form and compute natural gradients.
        3) Apply KL clipping to gradients.
        4) Perform an SGD-like parameter update.
        """
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']

        updates = {}
        # 1) Update eigen-decompositions every TInv steps
        for layer in self.modules:
            if self.steps % self.TInv == 0:
                self._update_inv(layer)
            # 2) Gather gradients in matrix form
            p_grad_mat = self._get_matrix_form_grad(layer)
            # 3) Compute natural gradients
            v = self._get_natural_grad(layer, p_grad_mat, damping)
            updates[layer] = v

        # 4) Apply KL clipping and update .grad
        self._kl_clip_and_update_grad(updates, lr)
        # 5) Perform the SGD-like step
        self._step_sgd(closure)

        self.steps += 1
