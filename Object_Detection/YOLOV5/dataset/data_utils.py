# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
from PIL import Image


def cvtColor(image):
    """将图像转换成RGB图像"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, letterbox_image):
    """对图像进行resize操作"""
    iw, ih = image.size
    w, h = size
    # 是否进行不失真的resize
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path):
    """获得类别数"""
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchors(anchors_path):
    """获得先验框"""
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def preprocess_input(image):
    """归一化处理"""
    image /= 255.0
    return image