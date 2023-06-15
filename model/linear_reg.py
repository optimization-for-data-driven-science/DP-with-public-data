import torch.nn as nn


class LinearRegModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.linear_reg_p, 1, bias=False)

    def forward(self, x):
        return self.linear(x)
