# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : mobilenetv3.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x+3)/6


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, activate):
        padding = (kernel_size - 1) // 2
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish()
        )


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channel, out_channel, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channel = in_channel // divide
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=mid_channel),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=out_channel),
            HardSwish(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = torch.flatten(out, start_dim=1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


class SEInverteBottleneck(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size, use_se, activate, stride):
        super(SEInverteBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channel == out_channel
        self.use_se = use_se

        self.conv = ConvBNActivation(in_channel=in_channel, out_channel=mid_channel, kernel_size=1, stride=1, groups=1, activate=activate)
        self.depth_conv = ConvBNActivation(in_channel=mid_channel, out_channel=mid_channel, kernel_size=kernel_size, stride=stride, groups=mid_channel, activate=activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(in_channel=mid_channel, out_channel=mid_channel)

        self.point_conv = ConvBNActivation(in_channel=mid_channel, out_channel=out_channel, kernel_size=1, stride=1, groups=1, activate=activate)

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        if self.use_shortcut:
            return x + out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(),
        )

        if self.type == 'large':
            self.large_bottleneck = nn.Sequential(
                SEInverteBottleneck(in_channel=16, mid_channel=16, out_channel=16, kernel_size=3, use_se=False, activate='relu', stride=1),
                SEInverteBottleneck(in_channel=16, mid_channel=64, out_channel=24, kernel_size=3, use_se=False, activate='relu', stride=2),
                SEInverteBottleneck(in_channel=24, mid_channel=72, out_channel=24, kernel_size=3, use_se=False, activate='relu', stride=1),
                SEInverteBottleneck(in_channel=24, mid_channel=72, out_channel=40, kernel_size=5, use_se=True, activate='relu', stride=2),
                SEInverteBottleneck(in_channel=40, mid_channel=120, out_channel=40, kernel_size=5, use_se=True, activate='relu', stride=1),
                SEInverteBottleneck(in_channel=40, mid_channel=120, out_channel=40, kernel_size=5, use_se=True, activate='relu', stride=1),
                SEInverteBottleneck(in_channel=40, mid_channel=240, out_channel=80, kernel_size=3, use_se=False, activate='hswish', stride=2),
                SEInverteBottleneck(in_channel=80, mid_channel=200, out_channel=80, kernel_size=3, use_se=False, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=80, mid_channel=184, out_channel=80, kernel_size=3, use_se=False, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=80, mid_channel=184, out_channel=80, kernel_size=3, use_se=False, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=80, mid_channel=480, out_channel=112, kernel_size=3, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=112, mid_channel=672, out_channel=112, kernel_size=3, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=112, mid_channel=672, out_channel=160, kernel_size=5, use_se=True, activate='hswish', stride=2),
                SEInverteBottleneck(in_channel=160, mid_channel=960, out_channel=160, kernel_size=5, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=160, mid_channel=960, out_channel=160, kernel_size=5, use_se=True, activate='hswish', stride=1),
            )
            self.large_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(960),
                HardSwish(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1, bias=False),
                HardSwish(),
            )
        else:
            self.small_bottleneck = nn.Sequential(
                SEInverteBottleneck(in_channel=16, mid_channel=16, out_channel=16, kernel_size=3, use_se=True, activate='relu', stride=2),
                SEInverteBottleneck(in_channel=16, mid_channel=72, out_channel=24, kernel_size=3, use_se=False, activate='relu', stride=2),
                SEInverteBottleneck(in_channel=24, mid_channel=88, out_channel=24, kernel_size=3, use_se=False, activate='relu', stride=1),
                SEInverteBottleneck(in_channel=24, mid_channel=96, out_channel=40, kernel_size=5, use_se=True, activate='hswish', stride=2),
                SEInverteBottleneck(in_channel=40, mid_channel=240, out_channel=40, kernel_size=5, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=40, mid_channel=240, out_channel=40, kernel_size=5, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=40, mid_channel=120, out_channel=48, kernel_size=5, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=48, mid_channel=144, out_channel=48, kernel_size=5, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=48, mid_channel=288, out_channel=96, kernel_size=5, use_se=True, activate='hswish', stride=2),
                SEInverteBottleneck(in_channel=96, mid_channel=576, out_channel=96, kernel_size=5, use_se=True, activate='hswish', stride=1),
                SEInverteBottleneck(in_channel=96, mid_channel=576, out_channel=96, kernel_size=5, use_se=True, activate='hswish', stride=1),
            )
            self.small_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(576),
                HardSwish(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1, bias=False),
                HardSwish(),
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_classes),
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

    def forward(self, x):
        x = self.first_conv(x)
        if self.type == 'large':
            x = self.large_bottleneck(x)
            x = self.large_last_stage(x)
        else:
            x = self.small_bottleneck(x)
            x = self.small_last_stage(x)

        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = MobileNetV3(num_classes=10)
    out = model(inputs)
    print(out.shape)