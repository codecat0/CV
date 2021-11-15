# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ssd_dataset.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from cv2 import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from data_utils import cvtColor, preprocess_input


class SSDDataset(Dataset):
    def __init__(
            self,
            annotation_lines,
            input_shape,
            anchors,
            batch_size,
            num_classes,
            train,
            overlap_threshold=0.5
    ):
        super(SSDDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        return len(self.annotation_lines)

    @staticmethod
    def rand(a=0, b=1):
        return np.random.rand()*(b - a) + a

    def get_random_data(
            self,
            annotation_line,
            input_shape,
            jitter=.3,
            hue=.1,
            sat=1.5,
            val=1.5,
            random=True
    ):
        line = annotation_line.split()
        # 读取图像并转换成RGB图像
        image = Image.open(line[0])
        image = cvtColor(image)
        # 获得图像的高宽与目标高宽
        iw, ih = image.size
        h, w = input_shape
        # 获得真实边界框及标签
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                bow_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_h>1, bow_w>1)]

            return image_data, box
        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = w / h + self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            new_h = int(scale * h)
            new_w = int(new_h * new_ar)
        else:
            new_w = int(scale * w)
            new_h = int(new_w / new_ar)

        image = image.resize((new_w, new_h), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w - new_w))
        dy = int(self.rand(0, h - new_h))
        new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * new_w / img_w + dx
            box[:, [1, 3]] = box[:, [1, 3]] * new_w / img_w + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def iou(self, box):
        """
        计算出当前真实框与所有的先验框的iou，判断真实框与先验框的重合情况
        """
        # 真实框与先验框重合部分的左上角和右下角
        inter_leftup = np.maximum(self.anchors[:, :2], box[:2])
        inter_rightbottom = np.minimum(self.anchors[:, 2:4], box[2:4])
        # 真实框与先验框重合部分的宽高、面积
        inter_wh = inter_rightbottom - inter_leftup
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_anchor = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        # 计算iou
        union = area_true + area_anchor - inter
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True, variances=(0.1, 0.1, 0.2, 0.2)):
        """
        计算当前真实框和先验框的重合情况
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        # 找到当前真实框重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold

        # 如果没有一个先验框重合度大于self.overlap_threshold，则选择重合度最大的为正样本
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        assigned_anchors = self.anchors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = assigned_anchors_center[:, 2:4] - assigned_anchors[:, 0:2]

        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]

        return encoded_box.ravel()

    def assigin_boxer(self, boxes):
        """
        assignment分为3个部分
            - :4   为网络应该有的回归预测结果
            - 4:-1 为先验框所对应的种类，默认为背景
            - -1   为当前先验框是否包含目标
        """
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # reshape后，获得的encoded_boxes的shape为：(num_true_box, num_anchors, 4+1)，4对应回归参数、1为iou
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
