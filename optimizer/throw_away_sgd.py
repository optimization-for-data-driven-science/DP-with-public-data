from opacus.optimizers import DPOptimizer
from typing import Callable, Optional
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
import torch


class ThrowAwaySgd(DPOptimizer):
    @classmethod
    def cast(cls, dp_optimizer: DPOptimizer):
        assert isinstance(dp_optimizer, DPOptimizer)
        dp_optimizer.__class__ = cls
        assert isinstance(dp_optimizer, ThrowAwaySgd)
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
        # # self.clip_and_accumulate()
        # public_batch_size = self.grad_samples[0].shape[0]
        # for p in self.params:
        #     _check_processed_flag(p.grad_sample)
        #     grad_sample = self._get_flat_grad_sample(p)
        #     grad = grad_sample.mean(dim=0)
        #     # print(grad.shape)
        #     if p.summed_grad is not None:
        #         p.summed_grad += grad
        #     else:
        #         p.summed_grad = grad
        #     _mark_as_processed(p.grad_sample)

        # if self._check_skip_next_step():
        #     self._is_last_step_skipped = True
        #     return False

        # # Remove Gaussian Noise because user data is public
        # # self.add_noise()
        self.scale_grad()

        # if self.step_hook:
        #     self.step_hook(self)

        # self._is_last_step_skipped = False

        # for p in self.params:
        #     p.grad = p.grad

        return True

    def step_public(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step_public():
            return self.original_optimizer.step()
        else:
            return None

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            # Don't do anything with private data
            return None
            # return self.original_optimizer.step()
        else:
            return None
