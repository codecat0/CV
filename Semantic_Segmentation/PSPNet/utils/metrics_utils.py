# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : metrics_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import csv
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threshold=0.5):
    """
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f-score = 2 * precision * recall / (precision + recall)
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), dim=-1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threshold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, dim=(0, 1))
    fp = torch.sum(temp_inputs, dim=(0, 1)) - tp
    fn = torch.sum(temp_target[..., :-1], dim=(0, 1)) - tp

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    precision = precision.mean()
    recall = recall.mean()
    score = 2 * precision * recall / (precision + recall)
    # score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # score = score.mean()
    return score


def get_confusion_matrix(target, predict, num_classes):
    """
        获得混淆矩阵，其中对角线上表示分类正确的像素点
    :param target: 转化为一维数组的标签，shape：(h, w)
    :param predict: 转化为一维数组的预测结果，shape: (h, w)
    :param num_classes: 类别数
    :return: shape：(num_classes, num_classes), 其中 (i, j) 表示 标签类别为i 预测结果为j的像素点的个数
    """
    mask = (target >= 0) & (target < num_classes)
    label = num_classes * target[mask].astype(int) + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix


def per_class_iou(confusion_matrix):
    """计算每个类别的iou"""
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, 1) + np.sum(confusion_matrix, 0) - np.diag(confusion_matrix)
    iou = intersection / np.maximum(union, 1)
    return iou


def per_class_recall(confusion_matrix):
    """
    计算每个类别的召回率: 某类别被预测正确的概率
    recall：tp / (tp + fn)
    """
    return np.diag(confusion_matrix) / np.maximum(confusion_matrix.sum(1), 1)


def per_class_precision(confusion_matrix):
    """
    计算每个类别的精准率: 某类别预测正确的概率
    precision = tp / (tp + fp)
    """
    return np.diag(confusion_matrix) / np.maximum(confusion_matrix.sum(0), 1)


def per_accuracy(confusion_matrix):
    """
    计算图像中像素准确率
    accuracy: (tp + tn) / (tp + tn + fp + fn)
    """
    return np.sum(np.diag(confusion_matrix)) / np.maximum(np.sum(confusion_matrix), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)

    confusion_matrix = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + '.png') for x in png_name_list]
    pred_imgs = [join(pred_dir, x + '.png') for x in png_name_list]

    for idx in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[idx]))
        label = np.array(Image.open(gt_imgs[idx]))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[idx], pred_imgs[idx]
                )
            )
            continue


        confusion_matrix += get_confusion_matrix(
            target=label.flatten(),
            predict=pred.flatten(),
            num_classes=num_classes
        )
        if idx > 0 and idx % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                idx,
                len(gt_imgs),
                100 * np.nanmean(per_class_iou(confusion_matrix)),
                100 * np.nanmean(per_class_recall(confusion_matrix)),
                100 * per_accuracy(confusion_matrix)
            ))

    IoUs = per_class_iou(confusion_matrix)
    Recall = per_class_recall(confusion_matrix)
    Precision = per_class_precision(confusion_matrix)

    for idx_classes in range(num_classes):
        print('===>' + name_classes[idx_classes] + ':\tIou-' + str(round(IoUs[idx_classes] * 100, 2))
              + '; Recall-' + str(round(Recall[idx_classes] * 100, 2)) + '; Precision-' + str(round(Precision[idx_classes] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_accuracy(confusion_matrix) * 100, 2)))
    return np.array(confusion_matrix, np.int32), IoUs, Recall, Precision


def adjust_axes(r, t, fig, axes):
    """调整绘图图像大小"""
    # 获取文本的尺寸
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    """绘制图像"""
    # 获得当前的Figure和Axes
    fig = plt.gcf()
    axes = plt.gca()

    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, confusion_matrix, IoUs, Recall, Precision, name_classes, tich_font_size=12):
    draw_plot_func(
        values=IoUs,
        name_classes=name_classes,
        plot_title="mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100),
        x_label="Intersection over Union",
        output_path=join(miou_out_path, 'mIoU.png'),
        tick_font_size=tich_font_size,
        plt_show=True
    )
    print("Save mIoU out to " + join(miou_out_path, 'mIoU.png'))

    draw_plot_func(
        values=Recall,
        name_classes=name_classes,
        plot_title="mRecall = {0:.2}%".format(np.nanmean(Recall) * 100),
        x_label="Recall",
        output_path=join(miou_out_path, 'Recall.png'),
        tick_font_size=tich_font_size,
        plt_show=False
    )
    print("Save Recall out to " + join(miou_out_path, 'Recall.png'))

    draw_plot_func(
        values=Precision,
        name_classes=name_classes,
        plot_title="mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
        x_label="Precision",
        output_path=join(miou_out_path, 'Precision.png'),
        tick_font_size=tich_font_size,
        plt_show=False
    )
    print("Save Precision out to " + join(miou_out_path, 'Precision.png'))

    with open(join(miou_out_path, 'confusion_matrix.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = list()
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(confusion_matrix)):
            writer_list.append([name_classes[i]] + [str(x) for x in confusion_matrix[i]])
        writer.writerows(writer_list)
    print('Save confusion_matrix out to ' + join(miou_out_path, 'confusion_matrix.csv'))