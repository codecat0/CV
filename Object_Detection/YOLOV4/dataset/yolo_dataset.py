# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : yolo_dataset.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from random import sample, shuffle
from cv2 import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from .data_utils import cvtColor, preprocess_input


class YoLoDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, mosaic, train):
        super(YoLoDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.mosaic = mosaic
        self.train = train

    def __getitem__(self, index):
        index = index % len(self.annotation_lines)
        if self.mosaic:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_mosaic(
                annotation_lines=lines,
                input_shape=self.input_shape
            )
        else:
            image, box = self.get_random_data(
                annotation_line=self.annotation_lines[index],
                input_shape=self.input_shape,
                is_image_enhance=self.train
            )
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def __len__(self):
        return len(self.annotation_lines)

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

    def get_random_data_with_mosaic(self, annotation_lines, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        # 确定分割线的位置
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)

        # 获取4个拼接图像拼接后的尺寸
        nws = [int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)),
               int(w * self.rand(0.4, 1))]
        nhs = [int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)),
               int(h * self.rand(0.4, 1))]

        # 获取4个拼接图像拼接时左上角的坐标，按逆时针方向拼接
        place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x),
                   int(w * min_offset_x)]
        place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y),
                   int(h * min_offset_y) - nhs[3]]

        image_datas = []
        box_datas = []

        for index, line in enumerate(annotation_lines):
            line_content = line.split()
            # 读取图像
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # 图片的宽、高
            iw, ih = image.size
            # 真实框的信息
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否水平翻转图像
            flip = self.rand() < .5
            if len(box) > 0 and flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            nw = nws[index]
            nh = nhs[index]
            image = image.resize((nw, nh), Image.BICUBIC)

            # 将图像放置在对应位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            # 对box进行处理
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / w + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / h + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图像拼接在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros((h, w, 3))
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 进行色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)

        x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # 对真实框进一步处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    @staticmethod
    def merge_bboxes(box_datas, cutx, cuty):
        merge_box = []
        for i, bboxes in enumerate(box_datas):
            for box in bboxes:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                # 左上角的图像
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty >= y1:
                        y2 = cuty
                    if x2 >= cutx >= x1:
                        x2 = cutx
                # 左下角
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty >= y1:
                        y1 = cuty
                    if x2 >= cutx >= x1:
                        x2 = cutx

                # 右下角
                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty >= y1:
                        y1 = cuty
                    if x2 >= cutx >= x1:
                        x1 = cutx

                # 右上角
                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty >= y1:
                        y2 = cuty
                    if x2 >= cutx >= x1:
                        x1 = cutx

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_box.append(tmp_box)

        return merge_box


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes