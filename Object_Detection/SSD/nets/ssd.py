# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ssd.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import vgg as add_vgg


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdims=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def add_extras(in_channels):
    layers = []
    # Conv8_2
    # 19x19x1024 -> 19x19x256 -> 10x10x512
    layers += [
        nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
    ]

    # Conv9_2
    # 10x10x512 -> 10x10x128 -> 5x5x256
    layers += [
        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
    ]

    # Conv10_2
    # 5x5x256 -> 5x5x128 -> 3x3x256
    layers += [
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
    ]

    # Conv11_2
    # 3x3x256 -> 3x3x128 -> 1x1x256
    layers += [
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
    ]

    return nn.ModuleList(layers)


class SSD300(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(SSD300, self).__init__()
        self.num_classe = num_classes
        self.vgg = add_vgg(pretrained)
        self.extras = add_extras(in_channels=1024)
        self.L2Norm = L2Norm(n_channels=512, scale=20)
        mbox = [4, 6, 6, 6, 4, 4]

        loc_layers = []
        conf_layers = []
        backbone_source = [21, -2]

        # VGG获得的特征图
        # 第21层和第-2层对应于Conv4_2和FC7
        # 可以用来作为回归预测和置信度预测
        for k, v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

        # 在add_extras获得的特征图
        # 第1、3、5、7层对应于Conv8_2,Conv9_2,Conv10_2,Conv11_2
        # 可以用来作为回归预测和置信度预测
        for k, v in enumerate(self.extras[1::2], start=2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

    def forward(self, x):
        source = list()
        loc = list()
        conf = list()

        # 获得Conv4_2的输出
        for i in range(23):
            x = self.vgg[i](x)

        # Conv4_2的输出需要进行L2标准化
        s = self.L2Norm(x)
        source.append(s)

        # 获得FC7的输出
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        source.append(x)

        # 获得extra层的输出
        # 第1、3、5、7层对应于Conv8_2,Conv9_2,Conv10_2,Conv11_2
        for i, m in enumerate(self.extras):
            x = F.relu(m(x), inplace=True)
            # 获得第1、3、5、7层的输出
            if i % 2 == 1:
                source.append(x)

        # 为获得的用来预测的6个特征图添加回归预测和置信度预测
        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 进行reshape堆叠
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)

        # loc: (bacth_size, num_default_box, 4)
        # conf: (batch_size, num_default_box, num_classes)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classe)
        )
        return output



if __name__ == '__main__':
    ssd = SSD300(num_classes=21)
    inputs = torch.randn(1, 3, 300, 300)
    output = ssd(inputs)
    print(output[0].shape)
    print(output[1].shape)