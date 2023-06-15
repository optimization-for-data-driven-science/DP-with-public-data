from opacus.optimizers import DPOptimizer
from typing import Callable, Optional
import torch


class PdaPdmdLinearOptimizer(DPOptimizer):
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
    def cast(cls, dp_optimizer: DPOptimizer, pub_mat_inv):
        assert isinstance(dp_optimizer, DPOptimizer)
        dp_optimizer.__class__ = cls
        assert isinstance(dp_optimizer, PdaPdmdLinearOptimizer)
        dp_optimizer.pub_mat_inv = pub_mat_inv
        w, v = torch.linalg.eig(pub_mat_inv)
        print(max(w.real))
        return dp_optimizer
    
    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        # print(self.params[0].grad)
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False

        # print(self.params[0].grad)

        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            # take linear weight parameter of linear model, ignore parameter b
            p = next(iter(self.params))
            private_grad = p.grad.clone()
            # new_private_grad = torch.linalg.solve(self.pub_mat_inv, private_grad.T)
            new_private_grad = torch.mm(self.pub_mat_inv, private_grad.T)
            new_private_grad = new_private_grad.T
            p.grad = new_private_grad
            return self.original_optimizer.step()
        else:
            return None
