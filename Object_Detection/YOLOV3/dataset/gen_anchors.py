# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : gen_anchors.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os
import json


class AnnotParser(object):
    def __init__(self, file_type):
        assert file_type in ['xml', 'json'], "Unsupported file type."
        self.file_type = file_type

    def parse(self, annot_dir):
        """
        解析标记文件
        :param annot_dir: 标记文件
        :return: shape: (n, 2)，每一行代表一个box，每一列代表归一化后box的宽高
        """
        if self.file_type == 'xml':
            return self.parse_xml(annot_dir)
        else:
            return self.parse_json(annot_dir)


    @staticmethod
    def parse_xml(annot_dir):
        """
        解析VOC数据集
        """
        boxes = []

        for xml_file in glob.glob(os.path.join(annot_dir, '*.xml')):
            tree = ET.parse(xml_file)

            h_img = int(tree.findtext('./size/height'))
            w_img = int(tree.findtext('./size/width'))

            for obj in tree.iter('object'):
                xmin = int(round(float(obj.findtext('bndbox/xmin'))))
                ymin = int(round(float(obj.findtext('bndbox/ymin'))))
                xmax = int(round(float(obj.findtext('bndbox/xmax'))))
                ymax = int(round(float(obj.findtext('bndbox/ymax'))))

                w_norm = (xmax - xmin) / w_img
                h_norm = (ymax - ymin) / h_img

                boxes.append([w_norm, h_norm])

        return np.array(boxes)

    @staticmethod
    def parse_json(annot_dir):
        """
        解析Coco数据集
        """
        boxes = []

        for js_file in glob.glob(os.path.join(annot_dir, '*.json')):
            with open(js_file) as f:
                data = json.load(f)

            h_img = data['imageHeight']
            w_img = data['imageWidth']

            for shape in data['shapes']:
                points = shape['points']
                xmin = int(round(points[0][0]))
                ymin = int(round(points[0][1]))
                xmax = int(round(points[1][0]))
                ymax = int(round(points[1][1]))

                w_norm = (xmax - xmin) / w_img
                h_norm = (ymax - ymin) / h_img

                boxes.append([w_norm, h_norm])

        return np.array(boxes)


class AnchorKmeans(object):
    """
    K-means 聚类生成anchors
    """
    def __init__(self, k, max_iter=1000, random_seed=None):
        self.k = k
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.n_iter = 0
        self.anchors_ = None
        self.labels_ = None
        self.ious_ = None

    def fit(self, boxes):
        """
        对输入的boxes进行K-means聚类
        :param boxes: shape(n, 2),其中2代表w和h
        """
        assert self.k < len(boxes), "K must be less than the number of data."

        if self.n_iter > 0:
            self.n_iter = 0

        np.random.seed(self.random_seed)
        n = boxes.shape[0]

        # 初始化k个聚类中心
        self.anchors_ = boxes[np.random.choice(n, self.k, replace=True)]
        # 初始化每个box的类别
        self.labels_ = np.zeros((n,))

        while True:
            self.n_iter += 1

            if self.n_iter > self.max_iter:
                break

            # 计算boxes和聚类中心的iou值，shape：(n, k)
            self.ious_ = self.iou(boxes, self.anchors_)
            distances = 1 - self.ious_
            cur_labels = np.argmin(distances, axis=1)

            # 如果聚类中心不再改变，跳出循环
            if (cur_labels == self.labels_).all():
                break

            # 更新聚类中心
            for i in range(self.k):
                self.anchors_[i] = np.mean(boxes[cur_labels == i], axis=0)

            self.labels_ = cur_labels

    @staticmethod
    def iou(boxes, anchors):
        """
        计算box和聚类中心anchor的iou值
        :param boxes: shape(n, 2)
        :param anchors: shape(k, 2)
        :return: shape(n, k)
        """

        # 这里利用广播机制
        w_min = np.minimum(boxes[:, None, 0], anchors[:, 0])
        h_min = np.minimum(boxes[:, None, 1], anchors[:, 1])
        inter = w_min * h_min

        box_area = boxes[:, 0] * boxes[:, 1]
        anchor_area = anchors[:, 0] * anchors[:, 1]
        union = box_area[:, np.newaxis] + anchor_area[np.newaxis]

        return inter / (union - inter)

    def avg_iou(self):
        """
        计算每个box所属类别的iou值的平均值
        """
        return np.mean(self.ious_[np.arange(len(self.labels_)), self.labels_])


if __name__ == '__main__':
    parser = AnnotParser(file_type='xml')
    boxes = parser.parse(annot_dir='../data/VOCdevkit/VOC2007/Annotations')
    print("Box num is: ", len(boxes))
    model = AnchorKmeans(k=9, random_seed=26)
    model.fit(boxes)
    print("Avg IOU is: ", model.avg_iou())
    print("The result anchors: \n", np.round(model.anchors_ * 416).astype(np.int32))