# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : alexnet.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""

import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 论文中的输出通道数为96，pytorch官方为64
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 论文中的输出通道数为256，pytorch官方为192
            # nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 这一操作是为了保证特征提取后的特征图大小为 6x6，使得网络可以接受224x224以外尺寸的图像
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # 提取图像特征
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        # 进行图像分类
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = alexnet(num_classes=10)
    inputs = torch.randn(1, 3, 224, 224)
    out = model(inputs)
    print(out.shape)
