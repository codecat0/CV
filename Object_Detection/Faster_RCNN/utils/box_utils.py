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
import numpy as np
import torch.nn.functional as F
from torchvision.ops import nms


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
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    # 防止分母为0或log出现负数情况
    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


class DecodeBox(object):
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes + 1

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou=0.3, confidence=0.5):
        results = []
        batch_size = len(roi_cls_locs)
        rois = rois.view((batch_size, -1, 4))
        # 对每一张图片进行处理
        for i in range(batch_size):
            roi_cls_loc = roi_cls_locs[i] * self.std
            roi_cls_loc = roi_cls_loc.view((-1, self.num_classes, 4))

            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox = cls_bbox.view((-1, self.num_classes, 4))

            cls_bbox[..., [0, 2]] = cls_bbox[..., [0, 2]] / input_shape[1] * image_shape[1]
            cls_bbox[..., [1, 3]] = cls_bbox[..., [1, 3]] / input_shape[0] * image_shape[0]

            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                c_confs = prob[:, c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )

                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c-1) * torch.ones((len(keep), 1))

                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
        return results
