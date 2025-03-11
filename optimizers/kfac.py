import math
import torch
import torch.optim as optim

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat

# based on https://github.com/alecwangcq/KFAC-Pytorch

class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.01,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=1e-1,
                 kl_clip=0.001,
                 weight_decay=0,
                 T_stats=10,
                 T_inv=100,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.T_stats = T_stats
        self.T_inv = T_inv

    def _save_input(self, module, input):
        #if not module.training: return None
        if torch.is_grad_enabled() and self.steps % self.T_stats == 0:
            aa = self.CovAHandler(input[0].data, module)
            #module._A = aa
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay, module, "A")

    def _save_grad_output(self, module, grad_input, grad_output):
        #if not module.training: return None
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.T_stats == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            #module._G = gg
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay, module, "G")

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.layers:
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            
            self.modules.append(module)
            module.register_forward_pre_hook(self._save_input)
            module.register_backward_hook(self._save_grad_output)
            print('(%s): %s' % (count, module))
            count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.linalg.eigh(
                self.m_aa[m], UPLO='U'
            )
        self.d_g[m], self.Q_g[m] = torch.linalg.eigh(
                self.m_gg[m], UPLO='U'
            )

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())
        
        d_a_inv = 1.0 / self.d_a[m]  # Inverse of eigenvalues for m_aa
        d_g_inv = 1.0 / self.d_g[m]  # Inverse of eigenvalues for m_gg

        # giving model access to inverses and log determinants 
        # not using this in current implementation
        #m.A_inv = self.Q_a[m] @ torch.diag(d_a_inv) @ self.Q_a[m].T  # Explicit inverse of m_aa
        #m.G_inv = self.Q_g[m] @ torch.diag(d_g_inv) @ self.Q_g[m].T
        #m.log_det_A = torch.sum(torch.log(self.d_a[m]))
        #m.log_det_G = torch.sum(torch.log(self.d_g[m]))


    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.q_mu.grad.data.view(m.q_mu.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.q_mu.grad.data
        #if m.q_bias_mu is not None:
        #    p_grad_mat = torch.cat([p_grad_mat, m.q_bias_mu.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.q_bias_mu is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.q_mu.grad.data.size())
            v[1] = v[1].view(m.q_bias_mu.grad.data.size())
        else:
            v = [v.view(m.q_mu.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            #if not m.training: continue
            v = updates[m]
            vg_sum += (v[0] * m.q_mu.grad.data * lr ** 2).sum().item()
            if m.q_bias_mu is not None:
                vg_sum += (v[1] * m.q_bias_mu.grad.data * lr ** 2).sum().item()
        nu = 1.0 if vg_sum == 0 else min(1.0, math.sqrt(self.kl_clip / vg_sum)) 

        for m in self.modules:
            #if not m.training: continue
            v = updates[m]
            m.q_mu.grad.data.copy_(v[0])
            m.q_mu.grad.data.mul_(nu)
            if m.q_bias_mu is not None:
                m.q_bias_mu.grad.data.copy_(v[1])
                m.q_bias_mu.grad.data.mul_(nu)

    def _step(self, closure):
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
                #print(param2name[p], p.grad.data)
                param_name = param2name.get(p, "<unknown>")
                # update to prior std
                if param_name == "p_log_sigma":
                    #p.data.add_(-group['lr'], d_p) NOTE: deprecated
                    p.data.add_(d_p, alpha=-group['lr'])
                    continue

                if weight_decay != 0 and self.steps >= 20 * self.T_stats:
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

                #p.data.add_(-group['lr'], d_p)
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
                self._update_inv(m)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1