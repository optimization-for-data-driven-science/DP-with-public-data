import torch.nn as nn


class EmnistConvV1(nn.Module):
    def __init__(self):
        super(EmnistConvV1, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=2)
        self.relu1 = nn.ReLU()

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=47)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        return x


