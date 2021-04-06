"""
Pytorch based CNN for MNIST dataset with fused batchnorm

http://yann.lecun.com/exdb/mnist/
"""

import torch
import torch.nn as nn
from .modules import ConvBN2d

__all__ = ['cnn_mnist_fused']

class Net(nn.Module):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        self.conv1 = ConvBN2d(in_channels=1, out_channels=8, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConvBN2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(576, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class cnn_mnist_fused:
    base = Net
    args = list()
    kwargs = {'num_class': 10}
