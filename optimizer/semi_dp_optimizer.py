from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, _generate_noise
from opt_einsum.contract import contract
from typing import Callable, Optional
import torch
import copy

class SemiDPOptimizer(DPOptimizer):

    @classmethod
    def cast(cls, dp_optimizer: DPOptimizer, beta, clip_norm_public=None):
        assert isinstance(dp_optimizer, DPOptimizer)
        dp_optimizer.__class__ = cls
        dp_optimizer.beta = beta
        dp_optimizer.clip_norm_public = clip_norm_public
        dp_optimizer.public_scale_grad = []
        assert isinstance(dp_optimizer, SemiDPOptimizer)
        return dp_optimizer

    def pre_step_public(
            self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        self.clip_and_accumulate()
            
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            p.grad = (p.summed_grad).view_as(p)
            _mark_as_processed(p.summed_grad)

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False

        self.public_scale_grad = []
        p = self.params[0]
        public_batch_size = p.grad_sample.shape[0]
        for p in self.params:
            p.grad /= public_batch_size
        for p in self.params:
            self.public_scale_grad.append(p.grad.clone().detach())
        return True

    def step_public(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        return self.pre_step_public()

    def pre_step_semi_dp(self):
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()

        self.scale_grad()
        for i, p in enumerate(self.params):
            with torch.no_grad():
                p.grad.copy_(self.beta * p.grad + (1 - self.beta) * self.public_scale_grad[i])

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step_semi_dp():
            return self.original_optimizer.step()
        else:
            return None
