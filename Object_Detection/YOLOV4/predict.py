# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : predict.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import time
import colorsys
from tqdm import tqdm

import numpy as np
from cv2 import cv2
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoLo
from dataset.data_utils import cvtColor, get_anchors, get_classes, preprocess_input, resize_image
from utils.bbox_utils import DecodeBox


class YoLoPredictor(object):
    _defaults = {
        'model_path': './model_data/yolo4_voc_weights.pth',
        'classes_path': './model_data/voc_classes.txt',
        'anchors_path': './model_data/yolo_anchors.txt',
        'anchors_mask': [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        'input_shape': (416, 416),
        'confidence': 0.3,
        'nms_iou': 0.3,
        'letterbox_image': False,
        'cuda': True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 获得种类、先验框的数量和box解码器
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.decode_box = DecodeBox(
            anchors=self.anchors,
            num_classes=self.num_classes,
            input_shape=self.input_shape,
            anchors_mask=self.anchors_mask
        )

        # 画框按照类别设置不同的颜色
        hsv_tuple = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuple))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 建立yolov3模型，载入yolov3模型的权重
        self.net = YoLo(
            anchors_mask=self.anchors_mask,
            num_classes=self.num_classes
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net.eval()
        print('{} model, anchors, and classes loaded'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()


    def detect_image(self, image):
        """
        检测图片
        :param image: PIL.Image格式
        """
        # 获取图像的h，w
        image_shape = np.array(np.shape(image)[0:2])
        # 将图像转换成RGB图像
        image = cvtColor(image)
        # 将图像resize到YOLOV3训练时输入大小
        image_data = resize_image(
            image=image,
            size=(self.input_shape[1], self.input_shape[0]),
            letterbox_image=self.letterbox_image
        )
        # 添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype=np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.decode_box.decode_box(outputs)
            # 将预测框进行NMS处理
            results = self.decode_box.non_max_suppression(
                prediction=torch.cat(outputs, dim=1),
                num_classes=self.num_classes,
                input_shape=self.input_shape,
                image_shape=image_shape,
                letterbox_image=self.letterbox_image,
                conf_threshold=self.confidence,
                nms_threshold=self.nms_iou
            )

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype=np.int32)
            top_conf = np.array(results[0][:, 4] * results[0][:, 5])
            top_boxes = results[0][:, :4]

        # 设置字体与边框厚度
        font = ImageFont.truetype(font='./model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1])//np.mean(self.input_shape), 2))

        # 图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle((left + i, top + i, right - i, bottom - i), outline=self.colors[c])
            draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)), fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.decode_box.decode_box(outputs)

            results = self.decode_box.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = self.decode_box.decode_box(outputs)
                results = self.decode_box.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.decode_box.decode_box(outputs)
            results = self.decode_box.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


if __name__ == '__main__':
    yolo_predictor = YoLoPredictor()
    mode = 'predict'
    video_path = 0
    video_save_path = ''
    video_fps = 25.0
    test_interval = 100
    dir_origin_path = './imgs'
    dir_save_path = 'imgs_out'

    if mode == 'predict':
        while True:
            img_path = input('Input image file path:')
            try:
                image = Image.open(img_path)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo_predictor.detect_image(image)
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
            frame = np.array(yolo_predictor.detect_image(frame))
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
                tact_time = yolo_predictor.get_FPS(image, test_interval)
                print(str(tact_time) + ' second, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # 遍历文件夹进行检测并保存
    elif mode == 'dir_predict':
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.ppm', '.tif', 'tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo_predictor.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")


