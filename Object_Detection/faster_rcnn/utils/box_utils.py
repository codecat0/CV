# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : box_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import math



def loc2bbox(src_bbox, loc, box_xform_clip=math.log(1000. / 16)):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], dim=-1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], dim=-1)
    src_center_x = torch.unsqueeze(src_bbox[:, 0], dim=-1) + 0.5 * src_width
    src_center_y = torch.unsqueeze(src_bbox[:, 1], dim=-1) + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    # 防止exp过大
    dw = torch.clamp(dw, max=box_xform_clip)
    dh = torch.clamp(dh, max=box_xform_clip)

    dst_center_x = dx * src_width + src_center_x
    dst_center_y = dy * src_height + src_center_y
    dst_width = torch.exp(dw) * src_width
    dst_height = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = dst_center_x - 0.5 * dst_width
    dst_bbox[:, 1::4] = dst_center_y - 0.5 * dst_height
    dst_bbox[:, 2::4] = dst_center_x + 0.5 * dst_width
    dst_bbox[:, 3::4] = dst_center_y + 0.5 * dst_height

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_center_x = src_bbox[:, 0] + 0.5 * src_width
    src_center_y = src_bbox[:, 1] + 0.5 * src_height

    dst_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_center_x = dst_bbox[:, 0] + 0.5 * dst_width
    dst_center_y = dst_bbox[:, 1] + 0.5 * dst_height

    # 防止分母为0或log中元素为负的情况
    eps = torch.finfo(src_height.dtype).eps
    src_width = torch.clamp(src_width, min=eps)
    src_height = torch.clamp(src_height, min=eps)

    dx = (dst_center_x - src_center_x) / src_width
    dy = (dst_center_y - src_center_y) / src_height
    dw = torch.log(dst_width / src_width)
    dh = torch.log(dst_height/ src_height)

    loc = torch.stack((dx, dy, dw, dh), dim=-1)
    return loc


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

def box_iou(bbox_a, bbox_b):
    area_a, area_b = bbox_area(bbox_a), bbox_area(bbox_b)

    left_top = torch.max(bbox_a[:, None, :2], bbox_b[:, :2])
    right_bottom = torch.min(bbox_a[:, None, 2:], bbox_b[:, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area_a[:, None] + area_b - inter)
    return iou