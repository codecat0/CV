# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : anchor_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np


class AnchorBox(object):
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=None):
        self.input_shape = input_shape
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)

    def call(self, layer_shape, mask=None):
        # 获取输入进来的特征层的宽和高
        layer_height = layer_shape[0]
        layer_width = layer_shape[1]

        # 获取输入进来的图片的宽和高
        img_height = self.input_shape[0]
        img_width = self.input_shape[1]

        box_widths = []
        box_heights = []

        # aspect_ratios
        # [1, 1, 2, 1/2] or [1, 1, 2, 1/2, 3, 1/3]
        for ar in self.aspect_ratios:
            # 首先添加一个较小的正方形
            if ar == 1 and len(box_widths) == 0:
                box_heights.append(self.min_size)
                box_widths.append(self.min_size)
            # 然后添加一个较大的正方形
            elif ar == 1 and len(box_widths) > 0:
                box_heights.append(np.sqrt(self.min_size * self.max_size))
                box_widths.append(np.sqrt(self.min_size * self.max_size))
            # 然后添加长方形
            elif ar != 1:
                box_heights.append(self.min_size / np.sqrt(ar))
                box_widths.append(self.min_size * np.sqrt(ar))

        # 获得所有先验框的1/2
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # 获取特征层对应的步长
        stride_x = img_width / layer_width
        stride_y = img_height / layer_height

        # 生成网格中心
        linx = np.linspace(0.5 * stride_x, img_width - 0.5 * stride_x, layer_width)
        liny = np.linspace(0.5 * stride_y, img_height - 0.5 * stride_y, layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        num_anchors_every_point = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_every_point))

        # 获得先验框的左上角和右下角
        anchor_boxes[:, ::4] -= box_widths
        anchor_boxes[:, 1::4] -= box_heights
        anchor_boxes[:, 2::4] += box_widths
        anchor_boxes[:, 3::4] += box_heights

        # 归一化
        anchor_boxes[:, ::2] /= img_width
        anchor_boxes[:, 1::2] /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)

        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
        return anchor_boxes


def get_output_length(height, width):
    """
    用于计算用于预测的特征层的大小
    """
    kernel_sizes = [2, 2, 2, 2, 3, 3, 3, 3]
    padding = [0, 0, 1, 0, 1, 1, 0, 0]
    stride = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_heights = []
    feature_widths = []
    for i in range(len(kernel_sizes)):
        height = (height + 2 * padding[i] - kernel_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - kernel_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)

    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]


def get_anchors(input_shape=(300, 300), anchor_size=(30, 60, 111, 213, 264, 315)):
    feature_heights, feature_widths = get_output_length(input_shape[0], input_shape[1])
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    anchors = []
    for i in range(len(feature_widths)):
        anchor_boxes = AnchorBox(
            input_shape=input_shape,
            min_size=anchor_size[i],
            max_size=anchor_size[i + 1],
            aspect_ratios=aspect_ratios[i]
        ).call((feature_heights[i], feature_widths[i]))
        anchors.append(anchor_boxes)
    anchors = np.concatenate(anchors, 0)
    return anchors
