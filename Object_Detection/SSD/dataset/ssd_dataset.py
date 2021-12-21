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
from .data_utils import cvtColor, preprocess_input


class SSDDataset(Dataset):
    def __init__(
            self,
            annotation_lines,
            input_shape,
            anchors,
            num_classes,
            train,
            overlap_threshold=0.5
    ):
        super(SSDDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.train = train
        self.overlap_threshold = overlap_threshold

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        index = index % len(self.annotation_lines)
        # image: 300x300x3; box: (num_box, 5)
        # 5: xmin, y_min, x_max, y_max, class_idx
        image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, is_image_enhance=self.train)
        # image: 3x300x300
        image_data = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        if len(box) != 0:
            # 取出真实框的位置信息
            boxes = np.array(box[:, :4], dtype=np.float32)
            # 归一化处理
            boxes[:, [0, 2]] /= self.input_shape[1]
            boxes[:, [1, 3]] /= self.input_shape[0]
            # 对真实框的种类进行one hot处理，
            one_hot_label = np.eye(self.num_classes-1)[np.array(box[:, 4], np.int32)]
            # 拼接后的形状为 (num_box, 4+20)
            box = np.concatenate([boxes, one_hot_label], axis=-1)

        box = self.assigin_boxer(box)
        return np.array(image_data), np.array(box)


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
            is_image_enhance=True
    ):
        # line 表示训练txt文件的某一行
        # 这一行的第一个表示图像的路径，后面的以(x_min, y_min, y_max, y_max, class_idx)表示一个真实框
        line = annotation_line.split()
        # 读取图片并转换为RGB格式
        image = Image.open(line[0])
        image = cvtColor(image)
        # 获取图像的宽，高
        img_w, img_h = image.size
        # 获取输入到SSD网络中图像的高，宽
        h, w = input_shape
        # 提取line后面的真实框，box shape：(num_box, 5)
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not is_image_enhance:
            # 将图像的尺寸缩放到输入到SSD网络中的尺寸
            scale = min(w / img_w, h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            dx = (w - new_w) // 2
            dy = (h - new_h) // 2

            # 将图像resize到指定尺寸
            image = image.resize((new_w, new_h), Image.BICUBIC)
            # 新创建一个图像，全灰色
            new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
            # 将图像粘贴到新建图像的中心
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 将图像的真实框进行对应的缩放，并限制其超出边界
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * new_w / img_w + dx
                box[:, [1, 3]] = box[:, [1, 3]] * new_h / img_h + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 滤除较小的框
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
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
        # iou shape: (num_anchors, ) 每个值代表每个anchor与当前box的iou值
        iou = self.iou(box)
        # encoded_box shape: (num_anchors, 4+1/0)
        # 4+1/0 : 前面4个表示真实要学习的回归参数，最后一个表示正样本anchor与当前真实框的iou值
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        # 找到当前真实框重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold

        # 如果没有一个先验框重合度大于self.overlap_threshold，则选择重合度最大的为正样本
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # 如果返回iou为True的话，将匹配到的正样本anchor的最后一个位置设置为iou值
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 获取正样本anchors
        assigned_anchors = self.anchors[assign_mask]
        # 计算真实框box的中心和宽高
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 计算先验框anchor的中心和宽高
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2]

        # 根据真实框和先验框计算每个先验框要学习的回归参数信息，variances表示对中心和宽高赋予不同权重
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]

        # 展平后返回，shape：(num_anchors * (4+1/0), )
        return encoded_box.ravel()

    def assigin_boxer(self, boxes):
        """
        assignment分为3个部分
            - :4   为网络应该有的回归预测结果
            - 4:-1 为先验框所对应的种类，默认为背景
            - -1   为当前先验框是否包含目标
        """
        # assignment shape: (num_anchors, 4+21+1)
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        # 初始时每个先验框anchor的背景概率为1
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算，并对每个真实框匹配到正样本进行编码处理(即计算其回归参数)
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # reshape后，获得的encoded_boxes的shape为：(num_true_box, num_anchors, 4+1)，4对应回归参数、1为iou
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        # 求取每一个先验框重合度最大的真实框
        # best_iou : (num_anchors,) 其值为anchor与box的最大iou值
        # best_iou_idx: (num_anchors,) 其值为anchor与box最大iou值box的索引
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # 保证每一个真实框都有对应的先验框进行匹配，求取每一个真实框重合度最大的先验框
        # box_iou: (num_true_box, ), 其值为box与anchor的最大iou值
        # box_iou_idx: (num_true_box,),其值为box与anchor最大iou值的anchor的索引
        box_iou = encoded_boxes[:, :, -1].max(axis=1)
        box_iou_idx = encoded_boxes[:, :, -1].argmax(axis=1)
        for i in range(len(box_iou_idx)):
            best_iou_idx[box_iou_idx[i]] = i
            best_iou[box_iou_idx[i]] = box_iou[i]

        # best_iou_mask: (num_anchors,) 其值是bool，最大iou值大于0为True，反之为False
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # 设置正样本的
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, best_iou_mask, :4]
        # 将正样本的背景概率设置为0
        assignment[:, 4][best_iou_mask] = 0
        # 设置正样本的标签
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        # 设置正样本含有物体
        assignment[:, -1][best_iou_mask] = 1
        return assignment


def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes