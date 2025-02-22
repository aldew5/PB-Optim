import torch
import torch.optim as optim
from models.bayes_linear import BayesianLinear
from utils.kfac_utils import *
import math


class NoisyKFAC(optim.Optimizer):
    def __init__(self, 
            model,
            N,
            lr=0.01, 
            damping=1e-1, 
            beta: float = 1e-2,
            weight_decay: float = 1e-4,
            momentum=0.9, # for updating kfactors
            lam: int = 0.5,
            kl_clip=0.001,
            eta= -1.6, #prior log std
            T_stats=10,
            T_inv=100,
            gamma_ex=1e-3,
            batch_averaged=True
        ):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.model = model
        self.damping = gamma_ex + lam/(N * eta)

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.steps = 0
        self.beta = beta

        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.modules = []
        self.batch_averaged = batch_averaged
        self.T_stats = T_stats
        self.T_inv = T_inv
        self.lam = lam
        self.N = N
        self.kl_clip = kl_clip

        # only for kfac-enabled BNNs
        #assert(model.kfac)
        self._prepare_model()

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.layers:
            self.modules.append(module)
            # hooks for getting activations and gradients for kfactors computation
            module.register_forward_pre_hook(self._save_input)
            module.register_backward_hook(self._save_grad_output)
            print('(%s): %s' % (count, module))
            count += 1

    def _save_input(self, module, input):
        """
        Use inputs to update A = aa^T with exp moving average.
        :param module: BayesianLinear layer
        :input: pre-layer activations
        """
        if torch.is_grad_enabled() and self.steps % self.T_stats == 0:
            aa = self.CovAHandler(input[0].data, module)
            # give the model access to kfactors
            module._A = aa
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.beta)

    def _save_grad_output(self, module, grad_input, grad_output):
        """
        Update G = gg^T with gradients hook. Will be called with a randomly sampled label
        :param module: Bayesialinear layer
        :grad_output: gradients for the layer activations
        """
       
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.T_stats == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # give the model access to the kfactors
            module._G = gg
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.beta)

    def _update_inv(self, layer, damping):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param layer: layer we are handling
        :param damping: damp the inverse calculation
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[layer], self.Q_a[layer] = torch.linalg.eigh(
                self.m_aa[layer], UPLO='U'
            )
        self.d_g[layer], self.Q_g[layer] = torch.linalg.eigh(
                self.m_gg[layer], UPLO='U'
            )

        self.d_a[layer].mul_((self.d_a[layer] > eps).float())
        self.d_g[layer].mul_((self.d_g[layer] > eps).float())

        # give model access to the inverse kroneckers for sampling
        # NOTE: we scale A_inv by lam/N here
        # TODO: damping here?
        layer.A_inv = self.lam/ self.N * self.Q_a[layer] @ torch.diag(1.0/(self.d_a[layer] + damping)) @ self.Q_a[layer].T
        layer.G_inv = self.Q_g[layer] @ torch.diag(1.0/(self.d_g[layer] + damping)) @ self.Q_g[layer].T



    def _kl_clip_and_update_grad(self, updates, lr):
        """
        :param updates: gradient (or natural gradient for weights)
        """
        # do kl clip
        vg_sum = 0

        # compute nu which scales the gradient
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weights.grad.data * lr ** 2).sum().item()
            #if m.q_bias_mu is not None:
            #    vg_sum += (v[1] * m.q_bias_mu.grad.data * lr ** 2).sum().item()
        nu = 1.0 if vg_sum == 0 else min(1.0, math.sqrt(self.kl_clip / vg_sum)) 
        
        # update grad with nu
        for m in self.modules:
            v = updates[m]
            m.weights.grad.data.copy_(v[0])
            m.weights.grad.data.mul_(nu)
            #if m.q_bias_mu is not None:
            #    m.q_bias_mu.grad.data.copy_(v[1])
            #    m.q_bias_mu.grad.data.mul_(nu)

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weights.grad.data.view(m.weights.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weights.grad.data
        #if m.q_bias_mu is not None:
        #    p_grad_mat = torch.cat([p_grad_mat, m.q_bias_mu.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        Note the modification from normal KFAC: we add a decay term to the matrix gradient.

        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ (p_grad_mat- self.lam * m.weights) @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        v = [v.view(m.weights.grad.data.size())]

        return v

    def _step(self, closure):
        """
        Compute gradients with momentum and update parameters
        """
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        param2name = {}
        for name, param in self.model.named_parameters():
            param2name[param] = name

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_name = param2name.get(p, "<unknown>")
                
                
                # update to prior std (ignore momentum)
                if param_name == "p_log_sigma":
                    #p.data.add_(-group['lr'], d_p) NOTE: deprecated
                    p.data.add_(d_p, alpha=-group['lr'])
                    continue
            
                # compute momentum grad for update
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        #buf.mul_(momentum).add_(1, d_p) NOTE: dep
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf

                #update param
                p.data.add_(d_p, alpha=-group['lr'])

 

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            #print(m)
            #if not m.training: continue
            classname = m.__class__.__name__
            if self.steps % self.T_inv == 0:
                self._update_inv(m, damping)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        # updates grad data to contain natural gradient
        self._kl_clip_and_update_grad(updates, lr)


        self._step(closure)
        self.steps += 1