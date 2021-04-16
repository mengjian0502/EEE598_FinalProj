"""
Pytorch based CNN for MNIST dataset

http://yann.lecun.com/exdb/mnist/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .binarize import BinaryConv2d, BinaryHardTanh, BinaryLinear, TerneryHardTanh 

__all__ = ['cnn_mnist_b']

class Net(nn.Module):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        self.conv1 = BinaryConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = TerneryHardTanh(inplace=False)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = BinaryLinear(in_features=864, out_features=num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class cnn_mnist_b:
    base = Net
    args = list()
    kwargs = {'num_class': 10}

if __name__ == '__main__':
    model_cfg = cnn_mnist_b
    net = cnn_mnist_b.base(*model_cfg.args, **model_cfg.kwargs).cuda()
    summary(net, (1,28,28))
