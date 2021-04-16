"""
Pytorch based CNN for MNIST dataset

http://yann.lecun.com/exdb/mnist/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .modules import QConv2d, QLinear

__all__ = ['cnn_mnist_q']

class Net(nn.Module):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        self.conv1 = QConv2d(in_channels=1, out_channels=16,kernel_size=3, stride=1, wbit=4, abit=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = QLinear(in_features=576, out_features=num_class, wbit=4, abit=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class cnn_mnist_q:
    base = Net
    args = list()
    kwargs = {'num_class': 10}

if __name__ == '__main__':
    model_cfg = cnn_mnist_q
    net = cnn_mnist_q.base(*model_cfg.args, **model_cfg.kwargs).cuda()
    summary(net, (1,28,28))
