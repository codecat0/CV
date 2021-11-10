# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from PIL import Image
import numpy as np


def cvtColor(image):
    """
    将图像转换成RGB图像
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2]:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    """
    对输入图片进行resize
    """
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    """
    图片归一化处理
    """
    image /= 255.0
    return image


def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width
