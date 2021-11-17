# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : box_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms


class BBoxUtility(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @staticmethod
    def ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    @staticmethod
    def decode_boxes(mbox_loc, anchors, variances):
        # 获得先验框的宽高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框中心的获取
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y

        # 真实框的宽高的获取
        decode_bbox_width = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width *= anchor_width
        decode_bbox_height = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height *= anchor_height

        # 获取真实框的左上角和右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        decode_bbox = torch.cat(
            (decode_bbox_xmin[:, None],
             decode_bbox_ymin[:, None],
             decode_bbox_xmax[:, None],
             decode_bbox_ymax[:, None]), dim=-1
        )

        decode_bbox = torch.clamp(decode_bbox, min=0, max=1)
        return decode_bbox

    def decode_box(
            self,
            predictions,
            anchors,
            image_shape,
            input_shape,
            letterbox_image,
            variances=(0.1, 0.2),
            nms_iou=0.3,
            confidence=0.5
    ):
        mbox_loc = predictions[0]
        mbox_conf = nn.Softmax(-1)(predictions[1])

        results = []

        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)
            for c in range(1, self.num_classes):
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )

                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c-1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c-1) * torch.ones((len(keep), 1))

                    c_pred = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()

                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy= (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2
                box_wh = results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return results