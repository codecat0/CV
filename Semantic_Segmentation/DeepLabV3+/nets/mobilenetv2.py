# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : mobilenetv2.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import math
import os

import torch
import torch.nn as nn
from torch.utils import model_zoo


def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride

        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw conv
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw conv
                nn.Conv2d(hidden_dim, out_channels, kernel_size=(1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                # pw conv
                nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw conv
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw conv
                nn.Conv2d(hidden_dim, out_channels, kernel_size=(1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channels = 32
        last_channels = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # (256, 256, 32) -> (256, 256, 16)
            [6, 24, 2, 2],  # (256, 256, 16) -> (128, 128, 24)
            [6, 32, 3, 2],  # (128, 128, 24) -> (64, 64, 32)
            [6, 64, 4, 2],  # (64, 64, 32) -> (32, 32, 64)
            [6, 96, 3, 1],  # (32, 32, 64) -> (32, 32, 96)
            [6, 160, 3, 2],  # (32, 32, 96) -> (16, 16, 160)
            [6, 320, 1, 1],  # (16, 16, 160) -> (16, 16, 320)
        ]

        assert input_size % 32 == 0
        input_channels = int(input_channels * width_mult)
        self.last_channels = int(last_channels * width_mult) if width_mult > 1.0 else last_channels
        self.features = [conv_bn(
            in_channels=3,
            out_channels=input_channels,
            stride=2
        )]

        for t, c, n, s in interverted_residual_setting:
            output_channels = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(
                            in_channels=input_channels,
                            out_channels=output_channels,
                            stride=s,
                            expand_ratio=t
                        )
                    )
                else:
                    self.features.append(
                        block(
                            in_channels=input_channels,
                            out_channels=output_channels,
                            stride=1,
                            expand_ratio=t
                        )
                    )
                input_channels = output_channels
        self.features.append(
            conv_1x1_bn(
                in_channels=input_channels,
                out_channels=self.last_channels
            )
        )
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channels, n_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url=url, model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model


if __name__ == '__main__':
    model = mobilenetv2()
    for i, layer in enumerate(model.features):
        print(i, layer)