import torch.nn as nn


class MNISTCvx(nn.Module):
    def __init__(self):
        super(MNISTCvx, self).__init__()

        # Convolutional layer 1
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.fc(x.flatten(start_dim=1))
        return x


