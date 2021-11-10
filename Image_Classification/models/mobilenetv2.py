# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : mobilenetv2.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""

import torch
import torch.nn as nn


class ConvBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            # 论文中提到，Relu6具有低精度计算时的鲁棒性
            nn.ReLU6(inplace=True)
        )


class InveredResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InveredResidual, self).__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise convolution
            layers.append(ConvBNReLu(in_channels, hidden_channels, kernel_size=1))
        layers.extend([
            # 3x3 depthwise convolution
            ConvBNReLu(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels),
            # 1x1 pointwise convolution (linear)
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        block = InveredResidual
        input_channels = 32
        last_channels = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 169, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLu(in_channels=3, out_channels=input_channels, stride=2))
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                # 每个bottleneck sequence的第一层的stride为s，其他层的stride的1
                stride = s if i == 0 else 1
                features.append(
                    block(in_channels=input_channels, out_channels=c, stride=stride, expand_ratio=t)
                )
                input_channels = c
        features.append(ConvBNReLu(in_channels=input_channels, out_channels=last_channels, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channels, num_classes)
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = MobileNetV2(num_classes=10)
    out = model(inputs)
    print(out.shape)
