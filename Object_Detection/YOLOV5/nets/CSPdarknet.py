# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : CSPdarknet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=1, 
                 stride=1, 
                 padding=None,
                 groups=1,
                 activation=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=autopad(kernel_size, padding),
            groups=groups,
            stride=stride,
            bias=False
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=0.001,
            momentum=0.03
        )
        self.act = SiLU() if activation else (activation if isinstance(activation, nn.Module) else nn.Identity())
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class Focus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 activation=True):
        super(Focus, self).__init__()
        self.conv = Conv(
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            activation=activation
        )

    def forward(self, x):
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], dim=1
            )
        )


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 groups=1,
                 expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(in_channels * expansion)
        self.cv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )
        self.cv2 = Conv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            groups=groups
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Module"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 number=1,
                 shortcut=True,
                 groups=1,
                 expansion=0.5):
        super(C3, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )
        self.cv2 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )
        self.cv3 = Conv(
            in_channels=2 * hidden_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.m = nn.Sequential(
            *[Bottleneck(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                shortcut=shortcut,
                groups=groups,
                expansion=1.0
            ) for _ in range(number)]
        )

    def forward(self, x):
        return self.cv3(
            torch.cat(
                [self.m(self.cv1(x)), self.cv2(x)], dim=1
            )
        )


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )
        self.cv2 = Conv(
            in_channels=hidden_channels * (len(kernel_sizes) + 1),
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2) for kernel_size in kernel_sizes]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat(
            [x] + [m(x) for m in self.m], dim=1
        ))


class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth):
        super(CSPDarknet, self).__init__()
        # (b, 3, 640, 640) -> (b, 64, 320, 320)
        self.stem = Focus(
            in_channels=3,
            out_channels=base_channels,
            kernel_size=3
        )

        # (b, 64, 320, 320) -> (b, 128, 160, 160)
        self.dark2 = nn.Sequential(
            Conv(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                kernel_size=3,
                stride=2
            ),
            C3(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                number=base_depth
            )
        )
        # (b, 128, 160, 160) -> (b, 256, 80, 80)
        self.dark3 = nn.Sequential(
            Conv(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
                kernel_size=3,
                stride=2
            ),
            C3(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                number=base_depth * 3
            )
        )
        # (b, 256, 80, 80) -> (b, 512, 40, 40)
        self.dark4 = nn.Sequential(
            Conv(
                in_channels=base_channels * 4,
                out_channels=base_channels * 8,
                kernel_size=3,
                stride=2
            ),
            C3(
                in_channels=base_channels * 8,
                out_channels=base_channels * 8,
                number=base_depth * 3
            )
        )
        # (b, 512, 40, 40) -> (b, 1024, 20, 20)
        self.dark5 = nn.Sequential(
            Conv(
                in_channels=base_channels * 8,
                out_channels=base_channels * 16,
                kernel_size=3,
                stride=2
            ),
            SPP(
                in_channels=base_channels * 16,
                out_channels=base_channels * 16
            ),
            C3(
                in_channels=base_channels * 16,
                out_channels=base_channels * 16,
                number=base_depth,
                shortcut=False
            )
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)

        x = self.dark3(x)
        feat1 = x

        x = self.dark4(x)
        feat2 = x

        x = self.dark5(x)
        feat3 = x

        return feat1, feat2, feat3