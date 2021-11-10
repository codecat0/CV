# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : vgg_roi_head.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn
from torchvision.ops import RoIPool


class VGG16RoIHead(nn.Module):
    def __init__(self, n_classes, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier

        self.roi = RoIPool(
            output_size=(roi_size, roi_size),
            spatial_scale=spatial_scale
        )

        # 对ROIPooling后的的结果进行回归预测
        self.cls_loc = nn.Linear(4096, n_classes * 4)
        # 对ROIPooling后的的结果进行分类
        self.score = nn.Linear(4096, n_classes)


    def forward(self, x, rois, roi_indices, img_size):
        n, _, h, w = x.shape
        if x.is_cuda:
            rois = rois.cuda()
            roi_indices = roi_indices.cuda()

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * w
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * h

        # shape: (anchors_num, 5) 5->batch_index, x1, y1, x2, y2
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)

        # 利用建议框对公用特征层进行截取并进行ROIPooling
        pool = self.roi(x, indices_and_rois)

        # 利用classifier网络进行特征提取
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))

        return roi_cls_locs, roi_scores
