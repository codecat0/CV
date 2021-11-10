# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : shufflenetv2.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn


class ConvBNReLu(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLu, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups):
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups),
            nn.BatchNorm2d(out_channel),
        )


class HalfSplit(nn.Module):
    """
    实现channel split
    """
    def __init__(self, dim=0, first_half=True):
        super(HalfSplit, self).__init__()
        self.first_half = first_half
        self.dim = dim

    def forward(self, x):
        splits = torch.chunk(x, 2, dim=self.dim)
        return splits[0] if self.first_half else splits[1]


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        # Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, dim0=1, dim1=2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x


class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channel, out_channel, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        if self.stride > 1:
            mid_channel = out_channel - in_channel
        else:
            mid_channel = out_channel // 2
            in_channel = mid_channel
            self.first_split = HalfSplit(dim=1, first_half=True)
            self.second_split = HalfSplit(dim=1, first_half=False)

        # 论文中Fig.3.(d) 中的右半部分
        self.bottleneck = nn.Sequential(
            # 1x1 Conv
            ConvBNReLu(in_channel=in_channel, out_channel=mid_channel, kernel_size=1, stride=1, groups=1),
            # 3x3 DWConv
            ConvBN(in_channel=mid_channel, out_channel=mid_channel, kernel_size=3, stride=stride, groups=mid_channel),
            # 1x1 Conv
            ConvBNReLu(in_channel=mid_channel, out_channel=mid_channel, kernel_size=1, stride=1, groups=1),
        )

        if self.stride > 1:
            # 论文中Fig.3.(d) 中的左半部分
            self.shortcut = nn.Sequential(
                # 3x3 DWConv
                ConvBN(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=stride, groups=in_channel),
                # 1x1 Conv
                ConvBNReLu(in_channel=in_channel, out_channel=in_channel, kernel_size=1, stride=1, groups=1),
            )

        self.channel_shuffle = ChannelShuffle(groups=groups)

    def forward(self, x):
        if self.stride > 1:
            x1 = self.bottleneck(x)
            x2 = self.shortcut(x)
        else:
            # channel split
            x1 = self.first_split(x)
            x2 = self.second_split(x)
            x1 = self.bottleneck(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_shuffle(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.groups = groups

        self.stage1 = nn.Sequential(
            ConvBNReLu(in_channel=3, out_channel=24, kernel_size=3, stride=2, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = self._make_layers(in_channel=24, out_channel=planes[0], block_num=layers[0], is_stage2=True)
        self.stage3 = self._make_layers(in_channel=planes[0], out_channel=planes[1], block_num=layers[1],
                                        is_stage2=False)
        self.stage4 = self._make_layers(in_channel=planes[1], out_channel=planes[2], block_num=layers[2],
                                        is_stage2=False)

        self.conv5 = ConvBNReLu(in_channel=planes[2], out_channel=planes[3], kernel_size=1, stride=1, groups=1)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=planes[3], out_features=num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, in_channel, out_channel, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channel=in_channel, out_channel=out_channel, stride=2,
                                      groups=1 if is_stage2 else self.groups))
        for _ in range(1, block_num):
            layers.append(
                ShuffleNetUnits(in_channel=out_channel, out_channel=out_channel, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv5(x)
        x = self.globalpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def shufflenet_v2_x0_5(**kwargs):
    planes = [48, 96, 192, 1024]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes=planes, layers=layers, groups=1, **kwargs)
    return model


def shufflenet_v2_x1_0(**kwargs):
    planes = [116, 232, 464, 1024]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes=planes, layers=layers, groups=1, **kwargs)
    return model


def shufflenet_v2_x1_5(**kwargs):
    planes = [176, 352, 704, 1024]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes=planes, layers=layers, groups=1, **kwargs)
    return model


def shufflenet_v2_x2_0(**kwargs):
    planes = [244, 488, 976, 2048]
    layers = [4, 8, 4]
    model = ShuffleNetV2(planes=planes, layers=layers, groups=1, **kwargs)
    return model


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = shufflenet_v2_x1_0(num_classes=10)
    out = model(inputs)
    print(out.shape)
