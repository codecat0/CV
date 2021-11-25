# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : predict.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import colorsys
import os
import time
from cv2 import cv2
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.faster_rcnn import FasterRCNN
from utils.data_utils import get_classes
from dataset.utils import cvtColor, get_new_img_size, resize_image, preprocess_input
from utils.box_utils import DecodeBox


class FRCNNPredictor(object):
    _defaults = {
        'model_path': './model_data/voc_weights_resnet.pth',
        'classes_path': './model_data/voc_classes.txt',
        'backbone': 'resnet50',
        'confidence': 0.5,
        'nms_iou': 0.3,
        'anchor_scales': (8, 16, 32)
    }

    @classmethod
    def get_defaults(cls, n):
        for n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name ' " + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if torch.cuda.is_available():
            self.std = self.std.cuda()
        self.decode_box = DecodeBox(self.std, self.num_classes)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 225)), self.colors))
        self.generate()

    def generate(self):
        """
        载入模型与权值
        """
        self.net = FasterRCNN(
            num_classes=self.num_classes,
            mode='predict',
            anchor_scales=self.anchor_scales,
            backbone=self.backbone
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        image = cvtColor(image)

        image_data = resize_image(image, (input_shape[1], input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if torch.cuda.is_available():
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)

            results = self.decode_box.forward(
                roi_cls_locs=roi_cls_locs,
                roi_scores=roi_scores,
                rois=rois,
                image_shape=image_shape,
                input_shape=input_shape,
                nms_iou=self.nms_iou,
                confidence=self.confidence
            )
            # 如果没有检测出物体，返回原图
            if len(results[0]) <= 0:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # 设置字体与边框厚度
        font = ImageFont.truetype(font='./model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thinckness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 2))

        # 绘制图像
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            left, top, right, bottom = box

            left = max(0, np.floor(left).astype('int32'))
            top = max(0, np.floor(top).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))

            label = '{} {:.3f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thinckness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image = cvtColor(image)
        image_data = resize_image(image, (image_shape[1], image_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)), 0)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                if torch.cuda.is_available():
                    images = images.cuda()

                roi_cls_locs, roi_scores, rois, _ = self.net(images)
                results = self.decode_box.forward(
                    roi_cls_locs=roi_cls_locs,
                    roi_scores=roi_scores,
                    rois=rois,
                    image_shape=image_shape,
                    input_shape=input_shape,
                    nms_iou=self.nms_iou,
                    confidence=self.confidence
                )
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, map_out_path):
        f = open(os.path.join(map_out_path, 'detection-results/' + image_id + '.txt'), 'w')
        image_shape = np.array(np.shape(image)[:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        image = cvtColor(image)
        image_data = resize_image(image, (image_shape[1], image_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if torch.cuda.is_available():
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)

            results = self.decode_box.forward(
                roi_cls_locs=roi_cls_locs,
                roi_scores=roi_scores,
                rois=rois,
                image_shape=image_shape,
                input_shape=input_shape,
                nms_iou=self.nms_iou,
                confidence=self.confidence
            )
            # 如果没有检测出物体，返回原图
            if len(results[0]) <= 0:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            left, top, right, bottom = box

            if predicted_class not in self.class_names:
                continue

            f.write(
                "%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom)))
            )

        f.close()
        return


if __name__ == '__main__':
    frcnn_predictor = FRCNNPredictor()
    # mode用于指定测试的模式：
    mode = 'predict'
    # video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    # video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    # video_fps用于保存的视频的fps
    video_path = "./imgs/v.mp4"
    video_save_path = "./imgs/detect_v.mp4"
    video_fps = 60.0
    # test_interval用于指定测量fps的时候，图片检测的次数
    test_interval = 100

    # dir_origin_path指定了用于检测的图片的文件夹路径
    # dir_save_path指定了检测完图片的保存路径
    dir_origin_path = 'img/'
    dir_save_path = 'img_out/'

    # 单张图片预测
    if mode == 'predict':
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = frcnn_predictor.detect_image(image)
                r_image.save('./img/1_dr.jpg')
                r_image.show()

    # 视频检测
    elif mode == 'video':
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(frcnn_predictor.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print('fps=%.2f' % fps)
            frame = cv2.putText(frame, 'fps= %.2f' % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('video', frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyWindow()

    # 测试fps
    elif mode == 'fps':
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                tact_time = frcnn_predictor.get_FPS(image, test_interval)
                print(str(tact_time) + ' second, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # 遍历文件夹进行检测并保存
    elif mode == 'dir_predict':
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.ppm', '.tif', 'tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = frcnn_predictor.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
