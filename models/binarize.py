"""
Binary quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import numpy as np

def Binarized(tensor):
    tensor = tensor.sign().add(0.1).sign()
    return tensor

def Quantize(tensor, nbit):
    assert nbit > 0., "the bit width must less than 0"
    
    if nbit > 0.9999:
        n = 2**nbit - 1
    else:
        n = nbit * 10 - 1

    out = tensor.clone()
    out = out.mul(n).round().div(n)
    return out 

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace, nbit):
        if inplace:
            ctx.mark_dirty(input)
        output = Quantize(input, nbit) 
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight Through Estimator
        """
        return grad_output, None, None

class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        absE = self.weight.org.abs().mean()
        self.weight.data = Binarized(self.weight.org)
        self.weight.data = self.weight.data.mul(absE)
        
        out = F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
        return out

class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        absE = self.weight.org.abs().mean()
        self.weight.data = Binarized(self.weight.org)
        self.weight.data = self.weight.data.mul(absE)
        out = F.linear(input, self.weight, self.bias)
        return out

class TerneryHardTanh(nn.Module):
    def __init__(self, inplace):
        super(TerneryHardTanh, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        input = torch.nn.functional.hardtanh(input, 0, 1)
        input = STE.apply(input, self.inplace, 0.3)
        return input

class BinaryHardTanh(nn.Module):
    def __init__(self, inplace):
        super(BinaryHardTanh, self).__init__()
        self.inplace = inplace
    
    def forward(self, input):
        input = F.hardtanh(input, 0, 1)
        input = STE.apply(input, self.inplace, 1)
        return input