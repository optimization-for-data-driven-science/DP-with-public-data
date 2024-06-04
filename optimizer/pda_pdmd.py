from opacus.optimizers import DPOptimizer
from typing import Callable, Optional
import torch
import numpy as np
import warnings


class PdaPdmdOptimizer(DPOptimizer):
    """
    Cast DPOptimizer to PDA-PDMD optimizer

    Args:
        dp_optimizer: original dp_optimizer returned by privacy_engine
        total_step: total_number of steps of this optimizer will execute
        K: hyperparameter of optimizer that control how fast ALPHA_T decays.
        ALPHA_T: default: None, if not None then use fixed value
            ALPHA_T instead of scheduling decaying ALPHA_T
            K will be ignored if ALPHA_T is not None
    """

    @classmethod
    def cast(cls, dp_optimizer: DPOptimizer, total_step, k, alpha_t=None):
        assert isinstance(dp_optimizer, DPOptimizer)
        dp_optimizer.__class__ = cls
        assert isinstance(dp_optimizer, PdaPdmdOptimizer)
        dp_optimizer.total_step = total_step
        dp_optimizer.K = k
        dp_optimizer.public_grad = []
        dp_optimizer.t = 0
        dp_optimizer.ALPHA_T = alpha_t
        return dp_optimizer

    def pre_step_public(
            self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        # In public settings, No need for noise
        # self.add_noise()
        # self.scale_grad()
        # Rewrite Scale function to match algorithm
        public_batch_size = len(self.grad_samples[0])
        for p in self.params:
            p.grad /= public_batch_size

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step_public(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step_public():
            self.public_grad = []
            for param in self.original_optimizer.param_groups:
                for p in param['params']:
                    if p.grad is None:
                        continue
                    self.public_grad.append(p.grad.clone())
        return None

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            cnt = 0
            if self.ALPHA_T is not None:
                ALPHA_T = self.ALPHA_T
            else:
                ALPHA_T = np.cos(np.pi * self.t / (self.K * self.total_step))
                if ALPHA_T <= 0:
                    warnings.warn(f"ALPHA_T {ALPHA_T} is less than 0. The range of ALPHA_T should be [0, 1]")
            for param in self.original_optimizer.param_groups:
                for p in param['params']:
                    if p.grad is None:
                        continue
                    p.grad = ALPHA_T * p.grad + (1 - ALPHA_T) * self.public_grad[cnt]
                    cnt += 1
            self.t += 1
            return self.original_optimizer.step()
        else:
            return None
