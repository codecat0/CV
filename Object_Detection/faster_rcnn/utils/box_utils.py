# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : box_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch



def loc2bbox(src_bbox, loc):
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

