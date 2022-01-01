# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : vggnet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch.nn as nn
from torch.hub import load_state_dict_from_url


model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

# 数字代表卷积操作，其值代表卷积核个数；M表示最大池化操作
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True)
            ]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg16(num_classes=1000, pretrained=False):
    model = VGG(features=make_layers(cfg), num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16_bn'], progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model