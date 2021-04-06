"""
Pytorch based CNN for MNIST dataset

http://yann.lecun.com/exdb/mnist/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['cnn_mnist']

class Net(nn.Module):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(576, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class cnn_mnist:
    base = Net
    args = list()
    kwargs = {'num_class': 10}
