# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : mobilenetv2.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'mobilenetv2': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar'
}


def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_channels = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise conv
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),

                # pointwise conv
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                # 升维
                nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),

                # depthwise conv
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),

                # pointwise conv
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channels = 32
        last_channels = 1280

        interverted_residual_settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        assert input_size % 32 == 0

        input_channels = int(input_channels * width_mult)
        self.last_channels = int(last_channels * width_mult) if width_mult > 1.0 else last_channels
        self.features = [conv_bn(3, input_channels, 2)]

        for t, c, n, s in interverted_residual_settings:
            output_channels = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channels, output_channels, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channels, output_channels, 1, expand_ratio=t))
                input_channels = output_channels

        self.features.append(conv_1x1_bn(input_channels, self.last_channels))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


def load_url(url, model_dir='./model_data', map_loaction=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_loaction)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)


def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(num_classes=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['mobilenetv2']), strict=False)
    return model