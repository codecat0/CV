# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : vgg.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
}


class VGG(nn.Module):
    def __init__(self, features, batch_norm, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.bacth_norm = batch_norm

    def forward(self, x):
        if self.bacth_norm:
            # (3, 512, 512) -> (64, 512, 512)
            feat1 = self.features[: 6](x)
            # (64, 512, 512) -> (128, 256, 256)
            feat2 = self.features[6: 13](feat1)
            # (128, 256, 256) -> (256, 128, 128)
            feat3 = self.features[13: 23](feat2)
            # (256, 128, 128) -> (512, 64, 64)
            feat4 = self.features[23: 33](feat3)
            # (512, 64, 64) -> (512, 32, 32)
            feat5 = self.features[33: -1](feat4)
        else:
            # (3, 512, 512) -> (64, 512, 512)
            feat1 = self.features[:4](x)
            # (64, 512, 512) -> (128, 256, 256)
            feat2 = self.features[4: 9](feat1)
            # (128, 256, 256) -> (256, 128, 128)
            feat3 = self.features[9: 16](feat2)
            # (256, 128, 128) -> (512, 64, 64)
            feat4 = self.features[16: 23](feat3)
            # (512, 64, 64) -> (512, 32, 32)
            feat5 = self.features[23: -1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels=3, batch_norm=False, **kwargs):
    model = VGG(make_layers(cfgs['D'], batch_norm=batch_norm, in_channels=in_channels), batch_norm=batch_norm, **kwargs)
    if pretrained:
        if batch_norm:
            state_dict = load_state_dict_from_url(model_urls['vgg16_bn'], model_dir='./model_data')
        else:
            state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model_data')
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model


if __name__ == '__main__':
    vgg16 = VGG16(pretrained=False, batch_norm=True, num_classes=2)
    inputs = torch.randn((1, 3, 512, 512))
    outs = vgg16(inputs)
    for out in outs:
        print(out.shape)