# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : yolo.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn

from .CSPdarknet import C3, Conv, CSPDarknet


class YoloBody(nn.Module):
    def __init__(self,
                 anchors_mask,
                 num_classes,
                 phi,
                 backbone='cspdarknet',
                 input_shape=(640, 640)):
        super(YoloBody, self).__init__()
        depth_dict = {
            's': 0.33,
            'm': 0.67,
            'l': 1.00,
            'x': 1.33
        }
        width_dict = {
            's': 0.50,
            'm': 0.75,
            'l': 1.00,
            'x': 1.25
        }
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.backbone_name = backbone
        if backbone == 'cspdarknet':
            self.backbone = CSPDarknet(
                base_channels=base_channels,
                base_depth=base_depth
            )
        else:
            raise ValueError('Backbone {} is not supported'.format(backbone))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_for_feat3 = Conv(
            in_channels=base_channels * 16,
            out_channels=base_channels * 8
        )
        self.conv3_for_upsample1 = C3(
            in_channels=base_channels * 16,
            out_channels=base_channels * 8,
            number=base_depth,
            shortcut=False
        )

        self.conv_for_feat2 = Conv(
            in_channels=base_channels * 8,
            out_channels=base_channels * 4
        )
        self.conv3_for_upsample2 = C3(
            in_channels=base_channels * 8,
            out_channels=base_channels * 4,
            number=base_depth,
            shortcut=False
        )

        self.down_sample1 = Conv(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            kernel_size=3,
            stride=2
        )
        self.conv3_for_downsample1 = C3(
            in_channels=base_channels * 8,
            out_channels=base_channels * 8,
            number=base_depth,
            shortcut=False
        )

        self.down_sample2 = Conv(
            in_channels=base_channels * 8,
            out_channels=base_channels * 8,
            kernel_size=3,
            stride=2
        )
        self.conv3_for_downsample2 = C3(
            in_channels=base_channels * 16,
            out_channels=base_channels * 16,
            number=base_depth,
            shortcut=False
        )

        self.yolo_head_P3 = nn.Conv2d(
            in_channels=base_channels * 4,
            out_channels=len(anchors_mask[2]) * (5 + num_classes),
            kernel_size=1
        )
        self.yolo_head_P4 = nn.Conv2d(
            in_channels=base_channels * 8,
            out_channels=len(anchors_mask[1]) * (5 + num_classes),
            kernel_size=1
        )
        self.yolo_head_P5 = nn.Conv2d(
            in_channels=base_channels * 16,
            out_channels=len(anchors_mask[0]) * (5 + num_classes),
            kernel_size=1
        )

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)

        # (b, 1024, 20, 20) -> (b, 512, 20, 20)
        P5 = self.conv_for_feat3(feat3)
        # (b, 512, 20, 20) -> (b, 512, 40, 40)
        P5_upsample = self.upsample(P5)
        # (b, 512, 40, 40) -> (b, 1024, 40, 40)
        P4 = torch.cat([P5_upsample, feat2], dim=1)
        # (b, 1024, 40, 40) -> (b, 512, 40, 40)
        P4 = self.conv3_for_upsample1(P4)
        # (b, 512, 40, 40) -> (b, 256, 40, 40)
        P4 = self.conv_for_feat2(P4)
        # (b, 256, 40, 40) -> (b, 256, 80, 80)
        P4_upsample = self.upsample(P4)
        # (b, 256, 80, 80) -> (b, 512, 80, 80)
        P3 = torch.cat([P4_upsample, feat1], dim=1)
        # (b, 512, 80, 80) -> (b, 256, 80, 80)
        P3 = self.conv3_for_upsample2(P3)

        # (b, 256, 80, 80) -> (b, 256, 40, 40)
        P3_downsample = self.down_sample1(P3)
        # (b, 256, 40, 40) -> (b, 512, 40, 40)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        # (b, 512, 40, 40) -> (b, 512, 40, 40)
        P4 = self.conv3_for_downsample1(P4)
        # (b, 512, 40, 40) -> (b, 512, 20, 20)
        P4_downsample = self.down_sample2(P4)
        # (b, 512, 20, 20) -> (b, 1024, 20, 20)
        P5 = torch.cat([P4_downsample, P5], dim=1)
        # (b, 1024, 20, 20) -> (b, 1024, 20, 20)
        P5 = self.conv3_for_downsample2(P5)

        # (b, 256, 80, 80) -> (b, 3*(5+num_classes), 80, 80)
        out2 = self.yolo_head_P3(P3)

        # (b, 512, 40, 40) -> (b, 3*(5+num_classes), 40, 40)
        out1 = self.yolo_head_P4(P4)

        # (b, 1024, 20, 20) -> (b, 3*(5+num_classes), 20, 20)
        out0 = self.yolo_head_P5(P5)

        return out0, out1, out2


if __name__ == '__main__':
    inputs = torch.randn((2, 3, 640, 640))
    model = YoloBody(
        anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        num_classes=20,
        phi='l'
    )
    outputs = model(inputs)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)

