# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dataset.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
from cv2 import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import cvtColor, preprocess_input


class FCNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(FCNDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 读取图像，jpg表示图像文件，png表示标签文件
        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, 'JPEGImages'), name + '.jpg'))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, 'SegmentationClass'), name + '.png'))

        # 数据增强处理
        jpg, png = self.get_random_data(jpg, png, self.input_shape, is_enhance=self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), (2, 0, 1))
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[png.reshape(-1)]
        seg_labels = seg_labels.reshape((int(self.input_shape[1]), int(self.input_shape[0]), self.num_classes + 1))

        return jpg, png, seg_labels

    @staticmethod
    def rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, is_enhance=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        h, w = input_shape

        if not is_enhance:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(scale * iw)
            nh = int(scale * ih)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # 对图像进行水平翻转操作
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 对图像进行平移操作
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # 对图像色彩进行调整
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] = 0
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, label


def fcn_dataset_collate(batch):
    """DataLoader中collate_fn使用"""
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels