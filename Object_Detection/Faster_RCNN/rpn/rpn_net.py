# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : rpn_net.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from utils.anchor_utils import enumerate_shifted_anchor, generate_base_anchor
from .proposal_create import ProposalGreator


class RegionProposalNetwork(nn.Module):
    def __init__(
            self,
            in_channels=512,
            mid_channels=512,
            ratios=(0.5, 1, 2),
            anchor_scales=(8, 16, 32),
            feature_map_stride=16,
            mode='training'
    ):
        super(RegionProposalNetwork, self).__init__()
        # 生成基础先验框, shape=(9, 4)
        self.base_anchor = generate_base_anchor(
            anchor_scales=anchor_scales,
            ratios=ratios
        )
        num_anchor = self.base_anchor.shape[0]

        # 先进行3x3的卷积
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)

        # 分类预测先验框内部是否包含物体
        self.score = nn.Conv2d(in_channels=mid_channels, out_channels=num_anchor * 2, kernel_size=1)

        # 回归预测先验框进行调整
        self.loc = nn.Conv2d(in_channels=mid_channels, out_channels=num_anchor * 4, kernel_size=1)

        # 图像下采样的步长，即特征图上点的间距
        self.feature_map_stride = feature_map_stride

        # 用于对建议框解码并进行非极大抑制
        self.proposal_layer = ProposalGreator(mode)


    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape

        # 先进行一个3x3的卷积，可理解为特征整合
        x = F.relu(self.conv1(x))

        # 分类预测先验框内部是否包含物体
        rpn_scores = self.score(x)
        # shape: (b, c, h, w) -> (b, h, w, c) -> (b, h * w * num_anchor, 2)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        # 进行softmax概率计算，每个先验框只有两个判别结果，
        # 内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        # 回归预测对先验框进行调整
        rpn_locs = self.loc(x)
        # shape: (b, c, h, w) -> (b, h, w, c) -> (b, h * w * num_anchor, 2)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # 生成先验框，此时获得的anchor是布满网格点的
        anchor = enumerate_shifted_anchor(
            base_anchor=self.base_anchor,
            feature_map_stride=self.feature_map_stride,
            height=h,
            width=w
        )

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                loc=rpn_locs[i],
                score=rpn_fg_scores[i],
                anchor=anchor,
                img_size=img_size,
                scale=scale
            )
            batch_idx = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_idx)

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor