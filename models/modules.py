"""
DNN quantization with / without merged modules
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import numpy as np
import torch.nn.init as init
from collections import OrderedDict


class WeightQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ub = 6
        lb = -3
        output_int = input.mul(scale[:,None,None,None]).round()
        output_int = output_int.clamp(lb, ub)                       # layer-wise clamp
        output_float = output_int.div(scale[:,None,None,None])
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        output_int = input.mul(scale).round_()
        output_float = output_int.div_(scale)
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class WQ(nn.Module):
    def __init__(self, wbit, num_features, channel_wise=1):
        super(WQ, self).__init__()
        self.wbit = wbit
        self.num_features = num_features

        if channel_wise:
            self.register_buffer('alpha_w', torch.ones(num_features))
        else:
            self.register_buffer('alpha_w', torch.tensor(1.))

        self.channel_wise = channel_wise

    def forward(self, input):
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
        z = z_typical[f'{int(self.wbit)}bit']
        n_lv = 2 ** (self.wbit - 1) - 1

        if self.channel_wise == 1:
            m = input.abs().mean([1,2,3])
            std = input.std([1,2,3])
            
            self.alpha_w = 1/z[0] * std - z[1]/z[0] * m
            lb = self.alpha_w.mul(-1.)
            ub = self.alpha_w

            # channel-wise clamp
            input = torch.max(torch.min(input, ub[:,None,None,None]), lb[:,None,None,None])
            scale = n_lv / self.alpha_w

            w_float = WeightQuant.apply(input, scale)
        else:
            m = input.abs().mean()
            std = input.std()
            
            self.alpha_w = 1/z[0] * std - z[1]/z[0] * m 
            input = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
            scale = n_lv / self.alpha_w

            w_float = RoundQuant.apply(input, scale)
        return w_float
    
    def extra_repr(self):
        return super(WQ, self).extra_repr() + 'wbit={}, channel_wise={}'.format(self.wbit, self.channel_wise)

class AQ(nn.Module):
    def __init__(self, abit, num_features, alpha_init):
        super(AQ, self).__init__()
        self.abit = abit
        self.alpha = nn.Parameter(torch.Tensor([alpha_init]))

    def forward(self, input):
        if input.size(1) > 3:
            input = torch.where(input < self.alpha, input, self.alpha)

            n_lv = 2**self.abit - 1
            scale = n_lv / self.alpha

            a_float = RoundQuant.apply(input, scale)
        else:
            a_float = input
        return a_float
    
    def extra_repr(self):
        return super(AQ, self).extra_repr() + 'abit={}'.format(self.abit)


class QConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False, 
        wbit=32, 
        abit=32, 
        channel_wise=0
    ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias
        )
        
        # precisions
        self.abit = abit
        self.wbit = wbit
        num_features = self.weight.data.size(0)

        self.WQ = WQ(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        self.AQ = AQ(abit=abit, num_features=num_features, alpha_init=10.0)
        
        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)

        out = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class QLinear(nn.Linear):
    r"""
    Fully connected layer with Quantized weight
    """
    def __init__(self, in_features, out_features, bias=True, wbit=32, abit=32, alpha_init=10.0):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # precisions
        self.wbit = wbit
        self.abit = abit
        self.alpha_init = alpha_init
        channels = self.weight.data.size(0)

        self.WQ = WQ(wbit=wbit, num_features=channels, channel_wise=0)
        self.AQ = AQ(abit=abit, num_features=channels, alpha_init=alpha_init)

        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)
        out = F.linear(input_q, weight_q, self.bias)
        return out