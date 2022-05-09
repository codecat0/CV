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
import torch
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input


class DeepLabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeepLabDataset, self).__init__()
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

        jpg, png = self.get_random_data(
            image=jpg,
            label=png,
            input_shape=self.input_shape,
            random=self.train
        )

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), (2, 0, 1))
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[png.reshape(-1)]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    @staticmethod
    def rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=.7, val=.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        #  对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw / ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        #  翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        #  高斯模糊
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #  旋转
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #  对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def deeplab_dataset_collate(batch):
    images, pngs, seg_labels = [], [], []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    # images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # pngs = torch.from_numpy(np.array(pngs)).long()
    # seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)

    return images, pngs, seg_labels


if __name__ == '__main__':
    data_path = '../../../data/VOCdevkit/VOC2007'
    with open(os.path.join(data_path, 'ImageSets/Segmentation/train.txt'), 'r') as f:
        train_lines = f.readlines()

    dataset = DeepLabDataset(
        annotation_lines=train_lines,
        input_shape=(512, 512),
        num_classes=20,
        train=True,
        dataset_path=data_path
    )

    for image, png, label in dataset:
        print(image.shape)
        print(png.shape)
        print(label.shape)
        break