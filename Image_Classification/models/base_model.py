# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : base_model.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch.nn as nn

from .alexnet import alexnet
from .vggnet import vgg11, vgg13, vgg16, vgg19
from .googlenet import GoogLeNet
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .densenet import densenet121, densenet169, densenet201
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3
from .shufflenetv1 import ShuffleNet
from .shufflenetv2 import ShuffleNetV2
from .ghostnet import GhostNet


class BaseModel(nn.Module):
    def __init__(self, name, num_classes):
        super(BaseModel, self).__init__()
        if name == 'alexnet':
            self.base = alexnet(num_classes=num_classes)
        elif name == 'vggnet':
            self.base = vgg16(num_classes=num_classes)
        elif name == 'googlenet':
            self.base = GoogLeNet(num_classes=num_classes)
        elif name == 'resnet':
            self.base = resnet34(num_classes=num_classes)
        elif name == 'densenet':
            self.base = densenet121(num_classes=num_classes)
        elif name == 'mobilenetv2':
            self.base = MobileNetV2(num_classes=num_classes)
        elif name == 'mobilenetv3':
            self.base = MobileNetV3(num_classes=num_classes)
        elif name == 'shufflenetv1':
            self.base = ShuffleNet(num_classes=num_classes)
        elif name == 'shufflenetv2':
            self.base = ShuffleNetV2(num_classes=num_classes)
        elif name == 'ghostnet':
            self.base = GhostNet(num_classes=num_classes)
        else:
            raise ValueError('Input model name is not supported!!!')

    def forward(self, x):
        return self.base(x)

