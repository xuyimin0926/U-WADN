import torch.nn as nn
from .deform_conv import DCN_layer, WADCN_layer
from .wac_op import *

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DGM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, width_mult_list):
        super(DGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = WADCN_layer(self.channels_in, self.channels_out, kernel_size, width_mult_list,
                                      padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out, width_mult_list)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter)
        out = dcn_out + sft_out
        out = x + out

        return out


class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out, width_mult_list):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            WidthAdaptiveConv2d(channels_in, channels_out, width_mult_list, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            WidthAdaptiveConv2d(channels_out, channels_out, width_mult_list, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            WidthAdaptiveConv2d(channels_in, channels_out, width_mult_list, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            WidthAdaptiveConv2d(channels_out, channels_out, width_mult_list, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta


class DGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, width_mult_list):
        super(DGB, self).__init__()

        # self.da_conv1 = DGM(n_feat, n_feat, kernel_size)
        # self.da_conv2 = DGM(n_feat, n_feat, kernel_size)
        self.dgm1 = DGM(n_feat, n_feat, kernel_size, width_mult_list)
        self.dgm2 = DGM(n_feat, n_feat, kernel_size, width_mult_list)
        self.conv1 = WidthAdaptiveConv2d(n_feat, n_feat, width_mult_list, kernel_size, padding=(kernel_size // 2), bias=True)
        self.conv2 = WidthAdaptiveConv2d(n_feat, n_feat, width_mult_list, kernel_size, padding=(kernel_size // 2), bias=True)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''

        out = self.relu(self.dgm1(x, inter))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dgm2(out, inter))
        out = self.conv2(out) + x

        return out


class DGG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks, width_mult_list):
        super(DGG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DGB(conv, n_feat, kernel_size, width_mult_list) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, width_mult_list, kernel_size, padding=(kernel_size // 2)))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        return res


class DGRN(nn.Module):
    def __init__(self, opt, conv=WidthAdaptiveConv2d, width_mult_list=[0.25, 0.5, 0.75, 1.0]):
        super(DGRN, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        channels = [int(n_feats * width_multi) for width_multi in width_mult_list]
        # head module
        modules_head = [conv([3 for _ in range(len(channels))], channels, width_mult_list, kernel_size, padding=(kernel_size // 2), bias=True)]
        self.head = nn.Sequential(*modules_head)

        # body
        modules_body = [
            DGG(conv, channels, kernel_size, n_blocks, width_mult_list) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(channels, channels, width_mult_list, kernel_size, padding=(kernel_size // 2), bias=True))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [conv(channels, [3 for _ in range(len(channels))], width_mult_list, kernel_size, padding=(kernel_size // 2), bias=True)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, inter):
        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x
