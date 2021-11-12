# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : faster_rcnn.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch.nn as nn

from backbones.vggnet import vgg16
# from backbones.resnet import resnet50
from backbones.resnet50 import resnet50
from rpn.rpn_net import RegionProposalNetwork
from roi_head.vgg_roi_head import VGG16RoIHead
from roi_head.resnet_roi_head import ResNet50RoIHead


class FasterRCNN(nn.Module):
    def __init__(
            self,
            num_classes,
            mode='training',
            feature_map_stride=16,
            anchor_scales=(8, 16, 32),
            ratios=(0.5, 1, 2),
            backbone='resnet50',
            pretrained=False
    ):
        super(FasterRCNN, self).__init__()
        self.feature_map_stride = feature_map_stride
        if backbone == 'vgg16':
            self.extractor, classifier = vgg16(
                pretrained=pretrained
            )
            self.rpn = RegionProposalNetwork(
                in_channels=512,
                mid_channels=512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feature_map_stride=self.feature_map_stride,
                mode=mode
            )
            self.head = VGG16RoIHead(
                n_classes=num_classes+1,
                roi_size=7,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(
                pretrained=pretrained
            )
            self.rpn = RegionProposalNetwork(
                in_channels=1024,
                mid_channels=512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feature_map_stride=self.feature_map_stride,
                mode=mode
            )
            self.head = ResNet50RoIHead(
                n_classes=num_classes+1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )
        else:
            raise ValueError("Input backbone name '{}' is Error, please check it again.".format(backbone))

    def forward(self, x, scale=1):
        # 计算输入图片的大小
        img_size = x.shape[2:]
        # 利用主干网络提取特征
        base_feature = self.extractor(x)
        # 获得建议框
        _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
        # 获得RoIHead的分类结果和回归结果
        roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)

        return roi_cls_locs, roi_scores, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(False)