# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : faster_rcnn_dataset.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from cv2 import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=(600, 600), train=True):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.train = train

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, item):
        image, target = self.get_random_data(
            annotation_line=self.annotation_lines[item],
            input_shape=self.input_shape,
            is_image_enhance=self.train
        )
        image = np.transpose(preprocess_input(image), (2, 0, 1))
        box_data = np.zeros((len(target), 5))
        if len(target) > 0:
            box_data[: len(target)] = target

        box = box_data[:, :4]
        label = box_data[:, -1]
        return image, box, label

    @staticmethod
    def rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def get_random_data(
            self,
            annotation_line,
            input_shape,
            jitter=0.3,
            hue=0.1,
            sat=1.5,
            val=1.5,
            is_image_enhance=True
    ):
        line = annotation_line.split()

        image = Image.open(line[0])
        image = cvtColor(image)

        img_w, img_h = image.size
        h, w = input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not is_image_enhance:
            scale = min(w / img_w, h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            dx = (w - new_w) // 2
            dy = (h - new_h) // 2

            image = image.resize((new_w, new_h), Image.BICUBIC)
            new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * new_w / img_w + dx
                box[:, [1, 3]] = box[:, [1, 3]] * new_h / img_h + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
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


def frcnn_dataset_collate(batch):
    images, bboxes, labels = [], [], []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)

    images = np.array(images)
    return images, bboxes, labels


if __name__ == '__main__':
    train_path = './model_data/train.txt'
    train_file = open(train_path)
    annotation_lines = [line.strip() for line in train_file.readlines()]
    dataset = FRCNNDataset(annotation_lines)
    data = dataset[26]
    print(data)