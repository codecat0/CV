# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : proposal_create.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
from torchvision.ops import nms

from utils.box_utils import loc2bbox


class ProposalGreator(object):
    def __init__(
            self,
            mode,
            nms_iou=0.7,
            n_train_pre_nms=12000,
            n_train_post_nms=600,
            n_test_pre_nms=3000,
            n_test_post_nms=300,
            min_size=16
    ):
        """
        :param mode: 设置预测or训练
        :param nms_iou: 建议框非极大抑制的iou大小
        :param n_train_pre_nms: 训练时NMS前用到的建议框数量
        :param n_train_post_nms: 训练时NMS后用到的建议框数量
        :param n_test_pre_nms: 预测时NMS前用到的建议框数量
        :param n_test_post_nms: 预测时NMS后用到的建议框数量
        :param min_size: 建议框的最小大小，小于时过滤掉
        """
        self.mode = mode
        self.nms_iou = nms_iou
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """
        :param loc: RPN网络回归预测结果
        :param score: RPN网络分类预测结果
        :param anchor: 先验框
        :param img_size: 图像尺寸
        :param scale: 缩放尺度
        """
        if self.mode == 'training':
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # 将先验框转换成tensor
        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()

        # 将RPN网络预测结果转化成建议框
        roi = loc2bbox(anchor, loc)

        # 防止建议框超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # 建议框的宽高的最小值不可以小于min_size * sclae
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]

        roi = roi[keep, :]
        score = score[keep]

        # 根据得分进行排序，取出NMS前的建议框
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]

        roi = roi[order, :]
        score = score[order]

        # 对建议框进行非极大抑制，取出NMS后的建议框
        keep = nms(roi, score, self.nms_iou)
        keep = keep[:n_post_nms]
        roi = roi[keep]

        return roi

