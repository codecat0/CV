# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loss_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)

    tmp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    tmp_target = target.view(-1)

    loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(tmp_inputs, tmp_target)
    return loss


def Focal_loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)

    tmp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    tmp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(tmp_inputs, tmp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = - ((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)

    tmp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    tmp_target = target.view(n, -1, ct)

    tp = torch.sum(tmp_target[..., :-1] * tmp_inputs, dim=(0, 1))
    fp = torch.sum(tmp_inputs, dim=(0, 1)) - tp
    fn = torch.sum(tmp_target[..., :-1], dim=(0, 1)) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + tp + smooth)
    loss = 1 - torch.mean(score)
    return loss