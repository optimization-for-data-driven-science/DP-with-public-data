from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, _generate_noise

import torch
from opt_einsum.contract import contract
from typing import Callable, Optional


class DPOptimizerDeepMind(DPOptimizer):

    @classmethod
    def cast(cls, dp_optimizer: DPOptimizer):
        assert isinstance(dp_optimizer, DPOptimizer)
        dp_optimizer.__class__ = cls
        assert isinstance(dp_optimizer, DPOptimizerDeepMind)
        return dp_optimizer

    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,))

        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(
                per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            # divide C here
            # must divide C before clipping, or it will no longer statisfy privacy constraints
            grad_sample.div_(self.max_grad_norm)

            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            # Original Code is std=self.noise_multiplier * self.max_grad_norm
            # Because we have already done the division before clipping,
            # Thus the noise is also scaled down by C.
            noise = _generate_noise(
                std=self.noise_multiplier,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            return self.original_optimizer.step()
        else:
            return None
