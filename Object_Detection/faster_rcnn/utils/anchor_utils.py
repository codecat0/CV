# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : anchor_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np


def generate_base_anchor(base_size=16, ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32)):
    """
    生成基础的先验框
    """
    base_anchor = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            base_anchor[index, 0] = - h / 2.
            base_anchor[index, 1] = - w / 2.
            base_anchor[index, 2] = h / 2.
            base_anchor[index, 3] = w / 2.
    return base_anchor



def enumerate_shifted_anchor(base_anchor, feature_map_stride, height, width):
    """
    对基础先验框进行拓展对应到特征图的所有特征点上
    """
    # 计算网格中心点
    shift_x = np.arange(0, width * feature_map_stride, feature_map_stride)
    shift_y = np.arange(0, height * feature_map_stride, feature_map_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 每个网格点生成生成9个先验框
    anchor_num_per_point = base_anchor.shape[0]
    num_point = shift.shape[0]
    anchor = base_anchor.reshape((1, anchor_num_per_point, 4)) + shift.reshape((num_point, 1, 4))

    # 所有先验框
    anchor = anchor.reshape((anchor_num_per_point * num_point), 4).astype(np.float32)

    return anchor


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nine_anchors = generate_base_anchor()
    print(nine_anchors)

    height, width, feature_map_stride = 38, 38, 16
    all_anchors = enumerate_shifted_anchor(nine_anchors, feature_map_stride, height, width)
    print(np.shape(all_anchors))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(-300, 900)
    plt.ylim(-300, 900)

    shift_x = np.arange(0, width * feature_map_stride, feature_map_stride)
    shift_y = np.arange(0, height * feature_map_stride, feature_map_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x, shift_y)
    box_widths = all_anchors[:, 2] - all_anchors[:, 0]
    box_heights = all_anchors[:, 3] - all_anchors[:, 1]
    k = 0
    for i in range(6000, 6009):
        if k % 3 == 0:
            rect = plt.Rectangle((all_anchors[i, 0], all_anchors[i, 1]), box_widths[i], box_heights[i], color='r', fill=False)
        elif k % 3 == 1:
            rect = plt.Rectangle((all_anchors[i, 0], all_anchors[i, 1]), box_widths[i], box_heights[i], color='g', fill=False)
        else:
            rect = plt.Rectangle((all_anchors[i, 0], all_anchors[i, 1]), box_widths[i], box_heights[i], color='b', fill=False)
        k = (k+1) % 3
        ax.add_patch(rect)
    plt.show()