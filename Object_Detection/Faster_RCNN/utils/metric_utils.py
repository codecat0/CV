# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : metric_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import glob
import json
import math
import operator
import os
import shutil
import sys

import numpy
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
    计算 log_avrerage miss rate、miss rate、false positive per image
    :param precision: (num_images, classes)
    :param fp_cumsum: (num_images, classes)
    :param num_images: int
    """
    if precision.size == 0:
        lamr, mr, fppi = 0, 1, 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(numpy.maximum(1e-10, ref))))

    return lamr, mr, fppi


def error(msg):
    """
    throw error and exit
    """
    print(msg)
    sys.exit(0)


def is_float_between_0_and_1(value):
    """
    check if the number is a float between 0.0 and 1.0
    """
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def voc_ap(rec, prec):
    """
    Calculate the AP given the recall and precision array
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mprec = prec[:]

    for i in range(len(mprec)-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mprec[i] != mprec[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mprec[i] - mprec[i-1]) * mprec[i]
    return ap, mrec, mprec


def file_lines_to_list(path):
    """
    Convert the lines of a file to a list
    """
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    """
    Draws text in image
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    thickness = 1
    bottem_left_corner_of_text = pos
    cv2.putText(
        img=img,
        text=text,
        org=bottem_left_corner_of_text,
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness
    )
    text_width, _ = cv2.getTextSize(
        text=text,
        fontFace=font,
        fontScale=font_scale,
        thickness=thickness)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    """
    Plot - adjust axes
    """
    # 为了重新缩放获取文本宽度，以inches为单位
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi

    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width

    x_lim = axes.get_xlim()
    axes.set_xlim(x_lim[0], x_lim[1] * propotion)


