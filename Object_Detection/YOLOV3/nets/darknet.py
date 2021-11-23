# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : darknet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from collections import OrderedDict
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x += residual
        return x


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.in_channels = 32
        # 3x416x416 -> 32x416x416
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.LeakyReLU(0.1)

        # 32x416x416 -> 64x208x208
        self.layer1 = self._make_layer(32, 64, layers[0])
        # 64x208x208 -> 128x104x104
        self.layer2 = self._make_layer(64, 128, layers[1])
        # 128x104x104 -> 256x52x52
        self.layer3 = self._make_layer(128, 256, layers[2])
        # 256x52x52 -> 512x26x26
        self.layer4 = self._make_layer(256, 512, layers[3])
        # 512x26x26 -> 1024x13x13
        self.layer5 = self._make_layer(512, 1024, layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

    def _make_layer(self, hidden_channels, out_channels, blocks):
        layers = list()
        # 下采样，步长为2，卷积核大小为3
        layers.append(
            ('ds_conv', nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(('ds_bn', nn.BatchNorm2d(out_channels)))
        layers.append(('ds_relu', nn.LeakyReLU(0.1)))
        self.in_channels = out_channels

        # 加入残差结构
        for i in range(blocks):
            layers.append(('residual_{}'.format(i), BasicBlock(self.in_channels, hidden_channels, out_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53():
    model = DarkNet(layers=(1, 2, 8, 8, 4))
    return model


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 416, 416)
    model = darknet53()
    out3, out4, out5 = model(inputs)
    # 2x256x52x52
    print(out3.shape)
    # 2x512x26x26
    print(out4.shape)
    # 2x1024x13x13
    print(out5.shape)
