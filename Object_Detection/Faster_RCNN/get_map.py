# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : get_map.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import time
import argparse
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from predict import FRCNNPredictor
from utils.metric_utils import get_map
from utils.data_utils import get_classes


def main(args):
    image_ids = open(os.path.join(args.voc_root, "ImageSets/Main/val.txt")).read().strip().split()

    if not os.path.exists(args.map_out_path):
        os.makedirs(args.map_out_path)
    if not os.path.exists(os.path.join(args.map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(args.map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(args.map_out_path, 'detection-results')):
        os.makedirs(os.path.join(args.map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(args.map_out_path, 'images-optional')):
        os.makedirs(os.path.join(args.map_out_path, 'images-optional'))

    class_names, _ = get_classes(args.classes_path)

    if args.map_mode == 0 or args.map_mode == 1:
        print('Load Model')
        faster_rcnn_predictor = FRCNNPredictor(confidence=0.01, nms_iou=0.5)
        print('Load Model done')

        print("Get predict result")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(args.voc_root, 'JPEGImages/' + image_id + '.jpg')
            image = Image.open(image_path)
            if args.map_vis:
                image.save(os.path.join(args.map_out_path, 'images-optional/' + image_id + '.jpg'))
            faster_rcnn_predictor.get_map_txt(image_id, image, args.map_out_path)
        print("Get predict result done")

    if args.map_mode == 0 or args.map_mode == 2:
        print("Get ground truth result")
        for image_id in tqdm(image_ids):
            with open(os.path.join(args.map_out_path, 'ground-truth/' + image_id + '.txt'), 'w') as new_f:
                root = ET.parse(os.path.join(args.voc_root, 'Annotations/' + image_id + '.xml')).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print('Get ground truth result done')

    if args.map_mode == 0 or args.map_mode == 3:
        print('Get map')
        get_map(args.min_overlap, True, args.map_out_path)
        print('Get map done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_root', type=str, default='../data/VOCdevkit/VOC2007')
    parser.add_argument('--map_out_path', type=str, default='map_out')
    parser.add_argument('--classes_path', type=str, default='./model_data/voc_classes.txt')
    parser.add_argument('--map_mode', type=int, default=0)
    parser.add_argument('--map_vis', action='store_false')
    parser.add_argument('--min_overlap', type=float, default=0.5)

    start = time.time()
    args = parser.parse_args()
    main(args)
    end = time.time()
    print('Get map consume {:.2f}s'.format(end-start))