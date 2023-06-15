from optimizer.semi_dp_optimizer import SemiDPOptimizer
from optimizer.pda_pdmd import PdaPdmdOptimizer
from optimizer.pda_pdmd_linear import PdaPdmdLinearOptimizer
from optimizer.dp_sgd_deepmind import DPOptimizerDeepMind
from optimizer.throw_away_sgd import ThrowAwaySgd
from torch.optim import Adam
from opacus.optimizers import DPOptimizer


class Dummy(DPOptimizer):
    @classmethod
    def cast(cls, dp_optimizer: DPOptimizer):
        return dp_optimizer


type2optimizer = {
    'semi_dp': SemiDPOptimizer,
    'pda_pdmd': PdaPdmdOptimizer,
    'pda_pdmd_linear': PdaPdmdLinearOptimizer,
    'deep_mind': DPOptimizerDeepMind,
    'throw_away_sgd': ThrowAwaySgd,
    'dp_sgd': Dummy,
    'adam': Adam,
    'semi_ldp': None,
    'ldp': None,
}

optimizer_choice = [key for key in type2optimizer.keys()]


def select_optimizer(args, pretrain=False):
    if pretrain:
        return type2optimizer['adam']
    optimizer = type2optimizer[args.optimizer]
    return optimizer
