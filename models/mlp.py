"""
Pytorch based MLP for MNIST dataset

http://yann.lecun.com/exdb/mnist/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mlp_mnist']

class MLP(nn.Module):
    def __init__(self, depth, num_class, dropout:bool=False, drop_rate:float=0.2):
        super(MLP, self).__init__()
        assert len(depth) > 0, "The configuration of the MLP cannot be empty"
        
        blocks = []
        in_channels = depth[0]
        for layer_idx in range(1,len(depth)):
            out_channels = depth[layer_idx]
            blocks += [
                nn.Linear(in_features=in_channels, out_features=out_channels),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                blocks += [nn.Dropout(p=drop_rate)]
            
            in_channels = out_channels
        
        self.mlp = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_channels, num_class, bias=True)   # last fc

    def forward(self, x):
        x = self.mlp(x)
        x = self.classifier(x)
        return x

class mlp_mnist:
    base = MLP
    args = list()
    kwargs = {'num_class': 10}

if __name__ == '__main__':
    x = torch.randn(28, 28)
    model = MLP([784, 400, 400], num_class=10, dropout=True)
    y = model(x)
    print(model)
    print(f"Size of the output = {y.size()}")