# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loss_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], label_smoothing=0):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.cuda = cuda

    @staticmethod
    def MSELoss(pred, target):
        return torch.pow(pred - target, 2)

    @staticmethod
    def BCELoss(pred, target):
        epsilon = 1e-7
        pred = torch.clip(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    @staticmethod
    def box_giou(b1, b2):
        """
        计算GIOU
        :param b1: (b, anchor_num, feat_h, feat_w, 4), xywh
        :param b2: (b, anchor_num, feat_h, feat_w, 4), xywh
        :return: (b, anchor_num, feat_h, feat_w)
        """
        # 计算预测框左上角和右下角
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        # 计算真实框左上角和右下角
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 计算真实框和预测框所有的iou
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        # 计算包含真实框与预测框的最小框的左上角和右下角
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    @staticmethod
    def smooth_labels(y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None, y_true=None):
        """
        :param l: 表示第几个有效特征层
        :param input: (b, 3*(5+num_classes), feat_h, feat_w)
        :param targets: (b, num_gt, 5)
        :param y_true: (b, 3, feat_h, feat_w, 5+num_classes)
        """
        # 获得图像数量，特征层的高和宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 计算步长
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 获得相对于特征层的anchor大小
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # (b, 3*(5+num_classes), feat_h, feat_w) -> (b, 3, 5+num_classes, feat_h, feat_w)
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 获取先验框的中心位置调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # 获取先验框的宽高调整参数
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])

        # 获取置信度
        conf = torch.sigmoid(prediction[..., 4])

        # 获取类别概率
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 将预测结果进行解码
        pred_boxes = self.get_pred_boxes(l, x, y, w, h, targets, scaled_anchors, in_h, in_w)

        if self.cuda:
            y_true = y_true.type_as(x)

        loss = 0
        n = torch.sum(y_true[..., 4] == 1)
        if n != 0:
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])
            loss_cls = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        else:
            tobj = torch.zeros_like(y_true[..., 4])
        loss_conf = torch.mean(self.BCELoss(conf, tobj))
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss

    def get_pred_boxes(self, l, x, y, w, h, targets, scaled_anchors, in_h, in_w):
        # 获取图像数量
        bs = len(targets)
        # 生成网格，先验框中心
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心和宽高
        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)
        return pred_boxes


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

