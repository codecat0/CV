# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : fcn.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn

from vggnet import vgg16
from resnet import resent34


class FCN(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=False, num_classes=21):
        super(FCN, self).__init__()
        self.backbone = backbone
        if backbone == 'resnet34':
            backbone = resent34(pretrained=pretrained)

            self.stage1 = nn.Sequential(*list(backbone.children())[:-5])    # 1/4
            self.stage2 = backbone.layer2   # 1/8
            self.stage3 = backbone.layer3   # 1/16
            self.stage4 = backbone.layer4   # 1/32

            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.upsample3 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.upsample4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=4, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )

            self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        elif backbone == 'vgg16':
            backbone = vgg16(pretrained=pretrained).features

            self.stage1 = nn.Sequential(*list(backbone.children())[:14])    # 1/4
            self.stage2 = nn.Sequential(*list(backbone.children())[14:24])  # 1/8
            self.stage3 = nn.Sequential(*list(backbone.children())[24:34])  # 1/16
            self.stage4 = nn.Sequential(*list(backbone.children())[34:44])  # 1/32

            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.upsample3 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.upsample4 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=6, stride=4, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        else:
            raise NotImplementedError("Unsupported backbone - '{}', Use vgg16 or resnet34".format(backbone))


    def forward(self, x):
        x1 = self.stage1(x)     # 1/4
        x2 = self.stage2(x1)    # 1/8
        x3 = self.stage3(x2)    # 1/16
        x4 = self.stage4(x3)    # 1/32

        out = self.upsample1(x4)    # 1/32 -> 1/16
        out = out + x3
        out = self.upsample2(out)   # 1/16 -> 1/8
        out = out + x2
        out = self.upsample3(out)   # 1/8 -> 1/4
        out = out + x1
        out = self.upsample4(out)   # 1/4 -> 1/1
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 512, 512)
    model = FCN()
    output = model(inputs)
    print(output.shape)