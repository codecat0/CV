# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : voc_annotations.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import argparse
import os
import random
import xml.etree.ElementTree as ET

from utils.data_utils import get_classes


def conver_annotation(voc_path, image_id, list_file, classes):
    in_file = open(os.path.join(voc_path, 'Annotations/{}.xml'.format(image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        box = (
            int(float(xmlbox.find('xmin').text)),
            int(float(xmlbox.find('ymin').text)),
            int(float(xmlbox.find('xmax').text)),
            int(float(xmlbox.find('ymax').text))
        )
        list_file.write(" " + ",".join([str(a) for a in box]) + ',' + str(cls_id))
    in_file.close()


def main(args):
    random.seed(0)
    if args.annotation_mode == 0 or args.annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xml_file_path = os.path.join(args.voc_path, 'Annotations')
        save_base_path = os.path.join(args.voc_path, 'ImageSets/Main')
        temp_xmls = os.listdir(xml_file_path)
        total_xmls = []
        for xml in temp_xmls:
            if xml.startswith('.xml'):
                total_xmls.append(xml)

        num_xmls = len(total_xmls)
        xml_idxs = range(num_xmls)
        trainval_num = int(num_xmls * args.trainval_percent)
        train_num = int(num_xmls * args.train_percent)
        trainval = random.sample(xml_idxs, trainval_num)
        train = random.sample(xml_idxs, train_num)

        print("train and val num is: ", trainval_num)
        print("train num is: ", train_num)

        f_trainval = open(os.path.join(save_base_path, 'trainval.txt'), 'w')
        f_test = open(os.path.join(save_base_path, 'test.txt'), 'w')
        f_train = open(os.path.join(save_base_path, 'train.txt'), 'w')
        f_val = open(os.path.join(save_base_path, 'val.txt'), 'w')

        for i in xml_idxs:
            name = total_xmls[i][:-4] + '\n'
            if i in trainval:
                f_trainval.write(name)
                if i in train:
                    f_train.write(name)
                else:
                    f_val.write(name)
            else:
                f_test.write(name)

        f_trainval.close()
        f_test.close()
        f_train.close()
        f_val.close()

        print("Generate txt in ImageSets done.")

    if args.annotation_mode == 0 or args.annotation_mode == 2:
        image_sets = ['train', 'val']
        classes, _ = get_classes(os.path.join(args.model_data, 'voc_classes.txt'))
        print("Generate train.txt and val.txt for train.")
        for image_set in image_sets:
            image_ids = open(os.path.join(args.voc_path, 'ImageSets/Main/{}.txt'.format(image_set)), encoding='utf-8').read().strip().split()
            list_file = open(os.path.join(args.model_data, '{}.txt'.format(image_set)), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write("{}/JPEGImages/{}.jpg".format(os.path.abspath(args.voc_path), image_id))
                conver_annotation(args.voc_path, image_id, list_file, classes)
                list_file.write('\n')
            list_file.close()

        print("Generate train.txt and val.txt for train done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--annotation_mode',
        type=int,
        default=2,
        help='0: 整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的train.txt、val.txt\
              1: 获得VOCdevkit/VOC2007/ImageSets里面的txt\
              2: 获得训练用的train.txt、val.txt'
    )
    parser.add_argument(
        '--voc_path',
        type=str,
        default='../../data/VOCdevkit/VOC2007',
        help='VOC数据集路径'
    )
    parser.add_argument(
        '--trainval_percent',
        type=float,
        default=0.9,
        help='指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1'
    )
    parser.add_argument(
        '--train_percent',
        type=float,
        default=0.9,
        help='指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1'
    )
    parser.add_argument(
        '--model_data',
        type=str,
        default='./model_data',
        help='类别文件路径以及生成train.txt、val.txt保存的路径'
    )

    args = parser.parse_args()
    print(args)
    main(args)