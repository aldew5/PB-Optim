from math import pow
from typing import Callable, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.optim
import torch.distributed as dist
from torch import Tensor


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVONPB(torch.optim.Optimizer):
    hessian_approx_methods = (
        'price',
        'gradsq',
    )

    def __init__(
        self,
        net,
        params, 
        lr: float,
        ess: float,
        delta: float, # pac-bayes bound confidence
        lam: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 0,
        mc_samples: int = 1,
        hess_approx: str = 'price',
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError(
                "Invalid number of MC samples: {}".format(mc_samples)
            )
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not 0.0 < hess_init:
            raise ValueError(
                "Invalid Hessian initialization: {}".format(hess_init)
            )
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(beta2))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

       
        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.net = net
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr
        self.delta = delta
        self.lam = lam
        self.param_name_map = {id(p): name for name, p in self.net.named_parameters()}


        # set initial temporary running averages
        self._reset_samples()
        # init all states
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        name2param = {param: name for name, param in self.net.named_parameters()}
        # only couns mus
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

    def _reset_samples(self):
        self.state['count'] = 0
        self.state['avg_grad'] = None
        self.state['avg_nxg'] = None
        self.state['avg_gsq'] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]

            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )
            group["hess"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            ).add(torch.as_tensor(hess_init))


            """
            # TODO: do I need to init these: how to init groups for partial_l2 and kl
            group['kl'] =torch.zeros(
                1, device=self._device, dtype=self._dtype
            )
            group["partial_l2"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )
            """


    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(
        self, train: bool, param_avg: Tensor, noise: Tensor
    ):
        param_grads = []
        offset = 0

        param_to_name = {}
        for name, param in self.net.named_parameters():
            param_to_name[param] = name

        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                #print(p.data)
                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.requires_grad:
                        #print(param_to_name[p])
                        param_grads.append(p.grad.flatten())
                    else:
                        #print(param_to_name[p])
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
            if self.hess_approx == 'price':
                self.state['avg_nxg'] = _welford_mean(
                    self.state['avg_nxg'], noise * grad_sample, count)
            elif self.hess_approx == 'gradsq':
                self.state['avg_gsq'] = _welford_mean(
                    self.state['avg_gsq'], grad_sample.square(), count)

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
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
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        dist.all_reduce(self.state["avg_nxg"])
        self.state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        noise_samples = []
        param_avgs = []
        param_to_name = {}
        for name, param in self.net.named_parameters():
            param_to_name[param] = name

        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=self._device, dtype=self._dtype)
                / (
                    group["ess"] * (group["hess"] + group["weight_decay"])
                ).sqrt()
            )
            noise_samples.append(noise_sample)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]

                param_avgs.append(p_avg)
                if "sigma" not in self.param_name_map.get(id(p), ""):
                    p.data = (p_avg + p_noise).view(p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]  # sanity check
        assert offset == self._numel  # sanity check

        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _update(self):
        self.current_step += 1
        
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])
            debias_factor = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0

            # Compute some constants (using your variables: ess, delta, kl, etc.)
            c1 = torch.log(torch.tensor(group['ess'] / self.delta))
            c2 = 2 * (group['ess'] - 1)
            l2 = torch.sqrt((self.state['kl'] + c1) / c2)

            # Flatten all parameters in this group into a single vector.
            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)

            #print(self.state["avg_grad"][pg_slice])
            # Update momentum using your provided method.
            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice], group["momentum"], b1
            )

            # m_bar for the entire group (debiased momentum).
            m_bar_full = group['momentum'] / debias_factor

           # print(m_bar_full)


            # Using the PRICE approximation for Hessian:
            if self.hess_approx == 'price' and self.state["avg_nxg"] is not None:
                price_term = self.state["avg_nxg"][pg_slice] * (group["ess"] * (group["hess"] + group["weight_decay"]))
            else:
                price_term = torch.zeros_like(m_bar_full)

            # Compute the s_hat term.
            s_hat = group["hess"]
            s_hat_slice = s_hat[pg_slice]
            s_hat_term = s_hat_slice / (2.0 * c2 * l2)

            # Combine terms to form g_s.
            g_s = s_hat_term + s_hat_slice + price_term

            # Update s_hat (Hessian approximation) for this group.
            denom = s_hat_slice + self.lam / group['ess']
            diff_sq = (g_s - s_hat_slice).square()
            new_s_hat_slice = (
                b2 * s_hat_slice +
                (1.0 - b2) * g_s +
                0.5 * (1.0 - b2)**2 * diff_sq / denom.clamp_min(1e-12)
            )
            s_hat[pg_slice] = new_s_hat_slice
            group["hess"] = s_hat

            # --- Extract only the mu parameters and corresponding momentum ---
            # We'll use a local offset for parameters within this group.
            group_offset = 0
            mu_indices = []  # list of tuples (start, end) for "mu" parameters in the flattened vector
            for p in group["params"]:
                numel = p.numel()
                # Look up the name using the parameter's id
                name = self.param_name_map.get(id(p), "")
                if "mu" in name:
                    mu_indices.append((group_offset, group_offset + numel))
                    
                group_offset += numel

            if mu_indices:
                # Extract the portions of param_avg and m_bar_full corresponding to mu parameters.
                mu_param_avg = torch.cat([param_avg[start:end] for start, end in mu_indices], dim=0)
                m_bar_mu = torch.cat([m_bar_full[start:end] for start, end in mu_indices], dim=0)
                # Also extract the corresponding portion of the new s_hat slice to form a denominator.
                denom2_mu = torch.cat([new_s_hat_slice[start:end] for start, end in mu_indices], dim=0) + (self.lam / group["ess"])
                # Compute the update vector for mu parameters.
                update_vec = (m_bar_mu + self.lam * mu_param_avg/(2 * c2* l2)) / denom2_mu.clamp_min(1e-12)
                update_vec = torch.clamp(update_vec, min=-group["clip_radius"], max=group["clip_radius"])

                # Now update only the mu entries in the overall parameter vector.
                new_param_avg = param_avg.clone()
                # We need to replace the slices corresponding to mu parameters.
                for start, end in mu_indices:
                    slice_len = end - start
                    new_param_avg[start:end] = param_avg[start:end] - lr * update_vec[:slice_len]
                    update_vec = update_vec[slice_len:]
            else:
                # If there are no "mu" parameters in this group, leave param_avg unchanged.
                new_param_avg = torch.sqrt(1/(group['ess'] * group['hess']))
                #new_param_avg = param_avg

            # Update the actual parameters in the group.
            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    numel = p.numel()
                    #print(self.param_name_map.get(id(p), ""), p.data)
                    p.data = new_param_avg[pg_offset:pg_offset + numel].view(p.shape)
                    pg_offset += numel
            assert pg_offset == group["numel"], "Mismatch in parameter update sizes."
            offset += group["numel"]
        assert offset == self._numel, "Mismatch in total parameter count."


    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == 'price':
            return avg_nxg[pg_slice] * hess
        elif method == 'gradsq':
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f'unknown hessian approx.: {method}')

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(
        method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd
    ) -> Tensor:
        f = IVONPB._get_nll_hess(
            method, hess + wd, avg_nxg, avg_gsq, pg_slice
        ) * ess
        return beta2 * hess + (1.0 - beta2) * f + \
            (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)

    @staticmethod
    def _new_param_averages(
        param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init
    ) -> Tensor:
        # momentum/debias = g bar / (1-beta_1^t)
        # this is the m update
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )