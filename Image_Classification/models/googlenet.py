# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : googlenet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""

import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

model_urls = {
    # GoogLeNet ported from TensorFlow
    "googlenet": "https://download.pytorch.org/models/googlenet-1378be20.pth",
}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        # 论文中没有batch_normaliztion，当时这个还没有提出
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, conv_block=None):
        """
        :param in_channels: 输入特征图的通道数
        :param ch1x1:       1x1卷积核的数量
        :param ch3x3red:    3x3卷积层前的1x1卷积核的数量
        :param ch3x3:       3x3卷积核的数量
        :param ch5x5red:    5x5卷积层前的1x1卷积核的数量
        :param ch5x5:       5x5卷积核的数量
        :param pool_proj:   最大池化层后1x1卷积核的数量
        :param conv_block:  卷积模块操作：conv -> bn -> relu
        """
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = conv_block(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            conv_block(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            # 论文中卷积核大小为 5x5，pytorch官方实现的是3x3
            # conv_block(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=3, padding=1)
            conv_block(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None, dropout=0.7):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.conv = conv_block(in_channels=in_channels, out_channels=128, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            aux_logits=False,
            transform_input=False,
            init_weights=True,
            blocks=None,
            dropout=0.2,
            dropout_aux=0.7
    ):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = conv_block(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32,
                                           pool_proj=32)
        self.inception3b = inception_block(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96,
                                           pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48,
                                           pool_proj=64)
        self.inception4b = inception_block(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64,
                                           pool_proj=64)
        self.inception4c = inception_block(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64,
                                           pool_proj=64)
        self.inception4d = inception_block(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64,
                                           pool_proj=64)
        self.inception4e = inception_block(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128,
                                           pool_proj=128)
        # 论文中采用的核大小为3x3，pytorch官方使用的2x2
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = inception_block(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128,
                                           pool_proj=128)
        self.inception5b = inception_block(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128,
                                           pool_proj=128)

        if aux_logits:
            self.aux1 = inception_aux_block(in_channels=512, num_classes=num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(in_channels=528, num_classes=num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        """
        标准化输入数据
        """
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7

        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, start_dim=1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x num_classes(1000)

        if self.aux_logits and self.training:
            return x, aux2, aux1
        else:
            return x


def googlenet(pretrained=False, progress=True, **kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False

        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['goooglenet'], progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        return model

    return GoogLeNet(**kwargs)


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = GoogLeNet(num_classes=10, aux_logits=False)
    out = model(inputs)
    print(out.shape)
