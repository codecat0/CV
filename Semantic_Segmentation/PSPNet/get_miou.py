# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : get_miou.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
from PIL import Image
from tqdm import tqdm

from predict import PSPNetPredictor
from utils.metrics_utils import compute_mIoU, show_results


if __name__ == '__main__':
    miou_mode = 0
    num_classes = 21
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    data_path = '../../data/VOCdevkit/VOC2007'

    image_ids = open(os.path.join(data_path, 'ImageSets/Segmentation/val.txt'), 'r').read().splitlines()
    gt_dir = os.path.join(data_path, 'SegmentationClass')
    miou_out_path = 'miou_out'
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # 获得预测结果
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print('Load model.')
        pspnet_predictor = PSPNetPredictor()
        print('Load model done.')

        print('Get predict result.')
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, 'JPEGImages/' + image_id + '.jpg')
            image = Image.open(image_path)
            image = pspnet_predictor.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + '.png'))
        print('Get predict result done.')

    # 计算miou
    if miou_mode == 0 or miou_mode == 2:
        print('Get miou.')
        confusion_matrix, IoUs, Recall, Precision = compute_mIoU(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            png_name_list=image_ids,
            num_classes=num_classes,
            name_classes=name_classes
        )
        print('Get miou done')
        show_results(
            miou_out_path=miou_out_path,
            confusion_matrix=confusion_matrix,
            IoUs=IoUs,
            Recall=Recall,
            Precision=Precision,
            name_classes=name_classes
        )

