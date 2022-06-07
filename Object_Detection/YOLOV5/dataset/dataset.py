# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dataset.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from random import sample, shuffle

from cv2 import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from .data_utils import cvtColor, preprocess_input
from .data_utils import get_classes, get_anchors


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes,
                 anchors, anchors_mask, mosaic, train, mosaic_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.mosaic = mosaic
        self.train = train
        self.mosaic_ratio = mosaic_ratio

        self.bbox_attrs = 5 + num_classes
        self.threshold = 4

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        index = index % len(self.annotation_lines)

        if self.mosaic:
            if self.rand() < self.mosaic_ratio:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[index])
                shuffle(lines)
                image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
            else:
                image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[0]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[1]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        y_true = self.get_target(box)
        return image, box, y_true

    @staticmethod
    def rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=.7, val=.4, random=True):
        line = annotation_line.split()

        # 读取图像并转换为RGB图像
        image = Image.open(line[0])
        image = cvtColor(image)

        # 获取图像的宽高和目标宽高
        iw, ih = image.size
        h, w = input_shape

        # 获得真实框
        box = np.array([
            np.array(list(map(int, box.split(',')))) for box in line[1:]
        ])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 对图像进行不失真的resize，并在多余部分以灰色覆盖
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 对真实框进行调整
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # 对图像进行缩放并进行长和宽的扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        # 对图像进行色域变换，计算色域变换参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像转换为HSv格式
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x - r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # 对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    @staticmethod
    def merge_bboxes(bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                # 左上角图像中的box
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty >= y1:
                        y2 = cuty
                    if x2 >= cutx >= x1:
                        x2 = cutx
                # 左下角图像中的box
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty >= y1:
                        y1 = cuty
                    if x2 >= cutx >= x1:
                        x2 = cutx
                # 右下角图像中的box
                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty >= y1:
                        y1 = cuty
                    if x2 >= cutx >= x1:
                        x1 = cutx
                # 左上角图像中的box
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
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=.7, val=.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            line_content = line.split()
            image = Image.open(line_content[0])
            image = cvtColor(image)
            iw, ih = image.size
            box = np.array(
                [np.array(list(map(int, box.split(',')))) for box in line_content[1:]]
            )
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            new_ar = iw / ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 左上角
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            # 左下角
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            # 右下角
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            # 左上角
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index += 1
            box_data = []

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                bow_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(bow_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[: len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros((h, w, 3))
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        # 对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像转换为HSV格式
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # 对框进行处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        return new_image, new_boxes

    @staticmethod
    def get_near_points(x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, targets):
        num_layers = len(self.anchors_mask)
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8, 3: 4}[l] for l in range(num_layers)]
        y_true = [
            np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32')
            for l in range(num_layers)
        ]
        box_best_ratio = [
            np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32')
            for l in range(num_layers)
        ]

        if len(targets) == 0:
            return y_true

        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]
            anchors = np.array(self.anchors) / {0: 32, 1: 16, 2: 8, 3: 4}[l]
            batch_target = np.zeros_like(targets)

            # 计算出真实框在特征层上的中心点
            batch_target[:, [0, 2]] = targets[:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[:, [1, 3]] * in_h
            batch_target[:, 4] = targets[:, 4]

            # 计算每一个真实框与每一个先验框的宽高的比值 (num_true_box, num_anchors, 2)
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            # 计算每一个先验框与每一个真实框的宽高的比值 (num_true_box, num_anchors, 2)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
            max_ratios = np.max(ratios, axis=-1)

            for t, ratio in enumerate(max_ratios):
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue

                    # 获得真实框属于哪一个网格点
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))

                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        # 若真实框的中心超出特征图范围，则跳过
                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            # 若之前存储的宽高比值大于当前宽高比值，则用当前的先验框匹配的真实框
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            # 若之前存储的宽高比值小于当前宽高比值，则用之前的先验框匹配的真实框
                            else:
                                continue

                        # 获取真实框的种类
                        c = int(batch_target[t, 4])
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1

                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
        return y_true


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    y_trues = [[] for _ in batch[0][2]]

    for img, box, y_true in batch:
        images.append(img)
        bboxes.append(box)
        for i, sub_y_true in enumerate(y_true):
            y_trues[i].append(sub_y_true)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
    return images, bboxes, y_trues


if __name__ == '__main__':
    train_annotation_path = '../model_data/train.txt'
    input_shape = (640, 640)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors_path = '../model_data/yolo_anchors.txt'
    classes_path = '../model_data/voc_classes.txt'

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    _, num_classes = get_classes(classes_path)
    anchors, _ = get_anchors(anchors_path)

    yolo_dataset = YoloDataset(
        annotation_lines=train_lines,
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        anchors_mask=anchors_mask,
        mosaic=True,
        train=True
    )

    image, box, y_true = yolo_dataset[0]
    print(image.shape)
    print(box.shape)
    for i in range(len(y_true)):
        print(y_true[i].shape)