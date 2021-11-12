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


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
    Draw plot using Matplotlib
    """
    sorted_dict_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dict_by_value)

    if true_p_bar != "":
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)

        plt.legend()

        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if (i + 1) == len(sorted_values):
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if (i+1) == len(sorted_values):
                adjust_axes(r, t, fig, axes)

    fig.canvas.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)

    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_inches = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_inches / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    plt.title(plot_title, fonsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()
    fig.savefig(output_path)
    if to_show:
        plt.show()
    plt.close()


def get_map(MINOVERLAP, draw_plot, path='./map_out'):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'image-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')

    show_animation = True
    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False

    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
    if not os.path.exists(RESULTS_FILES_PATH):
        os.makedirs(RESULTS_FILES_PATH)
    if draw_plot:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'AP'))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'F1'))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'Recall'))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'Precision'))
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, 'images', 'detections_one_by_one'))

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split('.txt', 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH, (file_id + '.txt'))
        if not os.path.exists(temp_path):
            error("Error: File not found: {}\n".format(temp_path))
        line_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in line_list:
            try:
                if 'difficult' in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                line_split = line.split()
                class_name = ""
                if 'difficult' in line:
                    _difficult = line_split[-1]
                    bottom = line_split[-2]
                    right = line_split[-3]
                    top = line_split[-4]
                    left = line_split[-5]
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name = class_name[:-1]
                    is_difficult = True
                else:
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]

            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append(
                    {
                        'class_name': class_name,
                        'bbox': bbox,
                        'used': False,
                        'difficult': True
                    }
                )
            else:
                bounding_boxes.append(
                    {
                        'class_name': class_name,
                        'bbox': bbox,
                        'used': False
                    }
                )

                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(TEMP_FILES_PATH, '/', file_id + '_ground_truth.json', 'w') as oufile:
            json.dump(bounding_boxes, oufile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split('.txt', 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + '.txt'))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error("Error: File not found: {}\n.".format(temp_path))
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append(
                        {
                            'confidence': confidence,
                            'file_id': file_id,
                            'bbox': bbox
                        }
                    )

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + " _dr.json", 'w') as oufile:
            json.dump(bounding_boxes, oufile)

    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(RESULTS_FILES_PATH + '/results.txt', 'w') as result_file:
        result_file.write("# AP and precision/recall per class\n")
        count_true_positive = {}

        for class_index, class_name in enumerate(gt_classes):
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            score05_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id = detection['file_id']
                score[idx] = float(detection['confidence'])
                if score[idx] > 0.5:
                    score05_idx = idx
                if show_animation:
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    if len(ground_truth_img) == 0:
                        error("Error: Image not found with id: " + file_id)
                    elif len(ground_truth_img) > 1:
                        error("Error: Multiple image with id: " + file_id)
                    else:
                        pass