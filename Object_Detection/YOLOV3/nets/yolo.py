# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : yolo.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from collections import OrderedDict
import torch
import torch.nn as nn

from .darknet import darknet53


def conv2d(in_channels, out_channels, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(channel_list, in_channels, out_channels):
    return nn.Sequential(
        conv2d(in_channels, channel_list[0], 1),
        conv2d(channel_list[0], channel_list[1], 3),
        conv2d(channel_list[1], channel_list[0], 1),
        conv2d(channel_list[0], channel_list[1], 3),
        conv2d(channel_list[1], channel_list[0], 1),
        conv2d(channel_list[0], channel_list[1], 3),
        nn.Conv2d(channel_list[1], out_channels, kernel_size=1, bias=True)
    )


class YoLo(nn.Module):
    def __init__(self, anchor_mask, num_classes):
        super(YoLo, self).__init__()
        self.backbone = darknet53()
        out_filters = self.backbone.layers_out_filters

        self.last_layer0 = make_last_layers(
            channel_list=[512, 1024],
            in_channels=out_filters[-1],
            out_channels=len(anchor_mask[-1]) * (num_classes + 5)
        )

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers(
            channel_list=[256, 512],
            in_channels=out_filters[-2] + 256,
            out_channels=len(anchor_mask[-2]) * (num_classes + 5)
        )

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers(
            channel_list=[128, 256],
            in_channels=out_filters[-3] + 128,
            out_channels=len(anchor_mask[-3]) * (num_classes + 5)
        )

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)

        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], dim=1)
        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)

        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], dim=1)
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2
