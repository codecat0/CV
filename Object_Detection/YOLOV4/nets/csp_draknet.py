# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : csp_draknet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish 激活函数： x * tanh(ln(1+exp(x))"""
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BasicConv(nn.Module):
    """CBM模块： Conv -> BN -> Mish"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size-1)//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Resblock(nn.Module):
    """残差单元"""
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


class Resblock_body(nn.Module):
    """CBM + CSP 结构"""
    def __init__(self, in_channels, out_channels, num_blocks, first=False):
        super(Resblock_body, self).__init__()
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)
        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        # (b, 3, 416, 416) -> (b, 32, 416, 416)
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            # (b, 32, 416, 416) -> (b, 64, 208, 208)
            Resblock_body(self.inplanes, self.feature_channels[0], layers[0], first=True),
            # (b, 64, 208, 208) -> (b, 128, 104, 104)
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1]),
            # (b, 128, 104, 104) -> (b, 256, 52, 52)
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2]),
            # (b, 256, 52, 52) -> (b, 512, 26, 26)
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3]),
            # (b, 512, 26, 26) -> (b, 1024, 13, 13)
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4])
        ])
        self.num_features = 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5


def darknet53(pretrained, **kwargs):
    model = CSPDarkNet(layers=[1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == '__main__':
    inputs = torch.randn((1, 3, 416, 416))
    model = darknet53(pretrained=False)
    o3, o4, o5 = model(inputs)
    print(o3.shape)
    print(o4.shape)
    print(o5.shape)