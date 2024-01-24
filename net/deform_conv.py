import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmcv.ops import modulated_deform_conv2d
from .wac_op import *

class DCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True, extra_offset_mask=True):
        super(DCN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.extra_offset_mask = extra_offset_mask
        # self.conv_offset_mask = nn.Conv2d(
        #     self.in_channels * 2,
        #     self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
        #     kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
        #     bias=True
        # )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # self.init_offset()
        # self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter):
        feat_degradation = torch.cat([input_feat, inter], dim=1)

        out = self.conv_offset_mask(feat_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(input_feat.contiguous(), offset, mask, self.weight, self.bias, self.stride,
                                       self.padding, self.dilation, self.groups, self.deformable_groups)



class WADCN_layer(DCN_layer):
    def __init__(self, in_channels_list, out_channels_list, kernel_size,
                 width_mult_list, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True, extra_offset_mask=True):
        super(WADCN_layer, self).__init__(max(in_channels_list), max(out_channels_list), kernel_size,
                                          stride=stride, padding=padding, dilation=dilation,
                                          groups=groups, deformable_groups=deformable_groups, bias=bias,
                                          extra_offset_mask=extra_offset_mask)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.width_mult_list = width_mult_list
        self.width_mult = 1.0

        self.conv_offset_mask = WidthAdaptiveConv2d(
            [channels * 2 for channels in in_channels_list],
            [self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] for _ in in_channels_list],
            width_mult_list, kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(max(out_channels_list)))
        else:
            self.register_parameter('bias', None)

        self.init_offset()
        self.reset_parameters()

    def forward(self, input_feat, inter):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        weight = self.weight[:self.in_channels, :self.out_channels, :, :]

        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias

        feat_degradation = torch.cat([input_feat, inter], dim=1)

        out = self.conv_offset_mask(feat_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(input_feat.contiguous(), offset, mask, weight, bias, self.stride,
                                       self.padding, self.dilation, self.groups, self.deformable_groups)
