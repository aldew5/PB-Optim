# @title Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import Tensor
from torch.optim.optimizer import _RequiredParameter as ClosureType
from typing import Optional


from models.bayes_linear import BayesianLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""
Doens't make sense to have hess because now we have hess per layer
"""

def kron_mv(A, G, v):
    # Efficient matrix-vector multiplication using the Kronecker product
    m, n = A.shape[0], G.shape[0]
    v = v.view(n, m)
    return (G @ v) @ A.t()

class KFACOptimizer(optim.Optimizer):
    def __init__(
        self, 
        params,
        model, 
        lr: float,
        ess: float, 
        damping=1e-1, 
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True
    ):
        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            ess=ess,
            clip_radius=clip_radius,
            damping=damping
        )

        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr
                
        self.model = model
        self.state = {}

    def _init_buffers(self):
        """

        """
        for group in self.param_groups:
            numel = group["numel"]

            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )

    def _update_kfac(A_old, A_new, b2):
        """
        Update hessian approximation matrices
        """
        if A_old is None:
            return A_new
        return b2 * A_old + (1-b2) * A_new


    def _compute_inverse_hess(self, layer):
        """
        Updates layer's Kronecker factors and returns inverse kronecker factors
        which can be used to update model parameters.
        """
        state = self.state

        A_old = state.get("A")
        G_old = state.get("G")
        A, G = layer._A, layer._G

        if A is not None and G is not None:
            A_new = self._update_kfac(
            A_old.detach() if A_old is not None else None,
            A.detach(),
            self.defaults['beta2']
            )
            G_new = self._update_kfac(
            G_old.detach() if G_old is not None else None,
            G.detach(),
            self.defaults['beta2'])
            
            # update layer's kronecker factors
            layer._A = A_new
            layer._G = G_new

            # compute inverses for inverse kfac approx
            A_inv = torch.inverse(A_new + self.defaults['damping'] * torch.eye(A_new.size(0)).to(A_new.device))
            G_inv = torch.inverse(G_new + self.defaults['damping'] * torch.eye(G_new.size(0)).to(G_new.device))
            return A_inv, G_inv
        else:
            return None
    
    def _restore_param_average(
        self, train: bool, param_avg: Tensor, noise: Tensor
    ):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.requires_grad:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel  # sanity check

        if train:  # collect grad sample for training
            grad_sample = torch.cat(param_grads, 0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(
                self.state["avg_grad"], grad_sample, count
            )
    
    def _update(self, layer):
        """
        Update model parameters
        """
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])
            A_inv, G_inv = self._compute_inverse_hess(layer)

            # param list
            param_avg = torch.cat(
                [p.flatten() for p in group["params"] if p is not None], 0
            )

            # update momentum using average gradient 
            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice], group["momentum"], b1
            )

    
            # update theta
            param_avg = self._new_param_averages(
                param_avg,
                group["momentum"],
                lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr,
                group["weight_decay"],
                group["clip_radius"],
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
                group["hess_init"]
            )

            # current slice offset in param list
            pg_offset = 0
            # use param_avg to update model params
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(
                        p.shape
                    )
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]  # sanity check
            offset += group["numel"]
        assert offset == self._numel  # sanity check

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for layer in self.model.modules():
            if isinstance(layer, BayesianLinear):
                A_inv, G_inv = self._compute_inverse_hess(layer)
                
                for name, param in layer.named_parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    
                    """
                    if name[2] == 'w': # weights
                        natural_grad = kron_mv(A_inv, G_inv, grad.view(-1))
                        state = self.state.setdefault(param, {})
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(natural_grad)

                        buf = state["momentum_buffer"]
                        buf.mul_(self.defaults['momentum']).add_(natural_grad)
                        param.data -= self.defaults['lr'] * buf.view_as(param)

                    elif name[2] == 'b': # biases
                        state = self.state.setdefault(param, {})
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(grad)

                        buf = state["momentum_buffer"]
                        buf.mul_(self.defaults['momentum']).add_(grad)
                        param.data -= self.defaults['lr'] * buf
                    else:
                        raise ValueError(f"Unknown parameter name: {name}")
                    """
                    self._update()

        self.model.p_log_sigma.data -= self.defaults['lr'] * self.model.p_log_sigma.grad

        return loss
    @torch.no_grad()
    def step2(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():  # explicit sync
            self._sync_samples()
        # update paramaters 
        self._update()
        self._reset_samples()
        return loss
    
    @staticmethod
    def _new_param_averages(
        param_avg, hess_inv, momentum, lr, wd, clip_radius, debias, hess_init
    ) -> Tensor:
        # update theta
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) * (hess_inv + wd),
            min=-clip_radius,
            max=clip_radius,
        )
    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad
    
