# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : resnet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""

import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False
    )


def conv1x1(in_channels, out_channels, stride=1):
    """
    1x1 convolution
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(in_channels=in_channels, out_channels=out_channels)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels, stride=stride)
        self.bn2 = norm_layer(out_channels)

        self.conv3 = conv1x1(in_channels=out_channels, out_channels=out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channels = 64
        assert len(layers) == 4

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        """
        :param block:       残差模块类型
        :param channels:    残差模块中3x3卷积核的数量
        :param blocks:      残差块的数目
        :param stride:      步长
        """
        norm_layer = self._norm_layer
        downsample = None

        # 当输入和输出维度不相同时，使用1x1卷积来匹配维度
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels=channels * block.expansion, stride=stride),
                norm_layer(channels * block.expansion)
            )

        layers = []
        layers.append(
            block(
                self.in_channels, channels, stride, downsample, norm_layer
            )
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels, channels, norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet(
        arch='resnet18',
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet(
        arch='resnet34',
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet(
        arch='resnet50',
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        arch='resnet101',
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet(
        arch='resnet152',
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = resnet34(num_classes=10)
    out = model(inputs)
    print(out.shape)
