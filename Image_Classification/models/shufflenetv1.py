# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : shufflenetv1.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn


class ConvBNRelU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups):
        padding = (kernel_size - 1) // 2
        super(ConvBNRelU, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channel, out_channel, groups):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(out_channel),
        )


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        # Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
        bacth_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        x = x.view(bacth_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, dim0=1, dim1=2).contiguous()
        x = x.view(bacth_size, -1, height, width)
        return x


class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channel, out_channel, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channel = out_channel - in_channel if self.stride > 1 else out_channel
        mid_channel = out_channel // 4

        self.bottleneck = nn.Sequential(
            # 1x1 GConv
            ConvBNRelU(in_channel=in_channel, out_channel=mid_channel, kernel_size=1, stride=1, groups=groups),
            # Channel Shuffle
            ChannelShuffle(groups=groups),
            # 3x3 DWConv
            ConvBNRelU(in_channel=mid_channel, out_channel=mid_channel, kernel_size=3, stride=stride, groups=mid_channel),
            # 1x1 GConv
            ConvBN(in_channel=mid_channel, out_channel=out_channel, groups=groups),
        )

        if self.stride > 1:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        if self.stride > 1:
            out = torch.cat([self.shortcut(x), out], dim=1)
        else:
            out += x
        return self.relu(out)


class ShuffleNet(nn.Module):
    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNet, self).__init__()

        self.stage1 = nn.Sequential(
            ConvBNRelU(in_channel=3, out_channel=24, kernel_size=3, stride=2, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage2 = self._make_layer(in_channel=24, out_channel=planes[0], groups=groups, block_num=layers[0], is_stage2=True)
        self.stage3 = self._make_layer(in_channel=planes[0], out_channel=planes[1], groups=groups, block_num=layers[1], is_stage2=False)
        self.stage4 = self._make_layer(in_channel=planes[1], out_channel=planes[2], groups=groups, block_num=layers[2], is_stage2=False)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=planes[2], out_features=num_classes)
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channel, out_channel, groups, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channel=in_channel, out_channel=out_channel, stride=2, groups=1 if is_stage2 else groups))
        for _ in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channel=out_channel, out_channel=out_channel, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.globalpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def shufflenet_g1(**kwargs):
    planes = [144, 288, 576]
    layers = [4, 8, 4]
    model = ShuffleNet(planes=planes, layers=layers, groups=1, **kwargs)
    return model


def shufflenet_g2(**kwargs):
    planes = [200, 400, 800]
    layers = [4, 8, 4]
    model = ShuffleNet(planes=planes, layers=layers, groups=2, **kwargs)
    return model


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = shufflenet_g1(num_classes=10)
    out = model(inputs)
    print(out.shape)