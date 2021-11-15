# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : vgg.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch.nn as nn
from torch.hub import load_state_dict_from_url

vgg16_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]


def vgg(pretrained=False):
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [
                nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    layers += [pool5]
    conv6 = [
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=6, padding=6),
        nn.ReLU(inplace=True)
    ]
    layers += conv6
    conv7 = [
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
        nn.ReLU(inplace=True)
    ]
    layers += conv7
    model = nn.ModuleList(layers)
    if pretrained:
        state_dict = load_state_dict_from_url(url=vgg16_url, model_dir='./model_data')
        state_dict = {k.replace('features.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    net = vgg()
    for i, layer in enumerate(net):
        print(i, layer)
