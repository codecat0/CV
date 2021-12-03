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
from .csp_draknet import darknet53


def conv2d(in_channels, out_channels, kernel_size, stride=1):
    """CBL结构"""
    pad = (kernel_size - 1) // 2
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.LeakyReLU(0.1)),
    ]))


class SpatialPyramidPooling(nn.Module):
    """SPP结构"""

    def __init__(self, pool_sizes=(5, 9, 13)):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes
        ])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features


class Upsample(nn.Module):
    """上采样"""

    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


def make_three_conv(filters_list, in_filters):
    """CBL x 3"""
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    """CBL x 5"""
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    """CBL + Conv"""
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoLo(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoLo, self).__init__()
        self.backbone = darknet53(pretrained=None)

        self.conv1 = make_three_conv([512, 1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(512, 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(256, 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

        self.down_sample1 = conv2d(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)
        self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)

        self.down_sample2 = conv2d(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)
        self.yolo_head1 = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        # (b, 1024, 13, 13) -> (b, 512, 13, 13)
        P5 = self.conv1(x0)
        # (b, 512, 13, 13) -> (b, 2048, 13, 13)
        P5 = self.SPP(P5)
        # (b, 2048, 13, 13) -> (b, 512, 13, 13)
        P5 = self.conv2(P5)

        # (b, 512, 13, 13) -> (b, 256, 26, 26)
        P5_upsample = self.upsample1(P5)
        # (b, 512, 26, 26) -> (b, 256, 26, 26)
        P4 = self.conv_for_P4(x1)
        # (b, 256, 26, 26) + (b, 256, 26, 26) -> (b, 512, 26, 26)
        P4 = torch.cat([P4, P5_upsample], dim=1)
        # (b, 512, 26, 26) -> (b, 256, 26, 26)
        P4 = self.make_five_conv1(P4)

        # (b, 256, 26, 26) -> (b, 128, 52, 52)
        P4_upsample = self.upsample2(P4)
        # (b, 256, 52, 52) -> (b, 128, 52, 52)
        P3 = self.conv_for_P3(x2)
        # (b, 128, 52, 52) + (b, 128, 52, 52) -> (b, 256, 52, 52)
        P3 = torch.cat([P3, P4_upsample], dim=1)
        # (b, 256, 52, 52) -> (b, 128, 52, 52)
        P3 = self.make_five_conv2(P3)

        # (b, 128, 52, 52) -> (b, 256, 26, 26)
        P3_downsample = self.down_sample1(P3)
        # (b, 256, 26, 26) + (b, 256, 26, 26) -> (b, 512, 26, 26)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        # (b, 512, 26, 26) -> (b, 256, 26, 26)
        P4 = self.make_five_conv3(P4)

        # (b, 256, 26, 26) -> (b, 512, 13, 13)
        P4_downsample = self.down_sample2(P4)
        # (b, 512, 13, 13) + (b, 512, 13, 13) -> (b, 1024, 13, 13)
        P5 = torch.cat([P4_downsample, P5], dim=1)
        # (b, 1024, 13, 13) -> (b, 512, 13, 13)
        P5 = self.make_five_conv4(P5)

        # (b, 128, 52, 52) -> (b, 3*(5+num_classes), 52, 52)
        out2 = self.yolo_head3(P3)

        # (b, 256, 26, 26) -> (b, 3*(5+num_classes), 26, 26)
        out1 = self.yolo_head2(P4)

        # (b, 512, 13, 13) -> (b, 3*(5+num_classes), 13, 13)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2


if __name__ == '__main__':
    inputs = torch.randn((1, 3, 416, 416))
    model = YoLo(anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], num_classes=20)
    o0, o1, o2 = model(inputs)
    print(o0.shape)
    print(o1.shape)
    print(o2.shape)
