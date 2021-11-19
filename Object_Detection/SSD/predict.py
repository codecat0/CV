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
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from nets.ssd import SSD300
from utils.anchor_utils import get_anchors
from utils.box_utils import BBoxUtility
from dataset.data_utils import cvtColor, get_classes, resize_image, preprocess_input


class SSDPredictor(object):
    _defaults = {
        'model_path': './model_data/ssd_weights.pth',
        'classes_path': './model_data/voc_classes.txt',
        'input_shape': (300, 300),
        'confidence': 0.3,
        'nms_iou': 0.25,
        'anchor_size': (30, 60, 111, 162, 213, 264, 315),
        'letterbox_image': False,
        'cuda': False
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

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = torch.from_numpy(get_anchors(
            input_shape=self.input_shape,
            anchor_size=self.anchor_size
        )).type(torch.FloatTensor)
        self.num_classes += 1

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.bbox_util = BBoxUtility(self.num_classes)
        self.generate()

    def generate(self):
        self.net = SSD300(self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net.eval()
        print('{} model, anchors, and classes load.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()


    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            results = self.bbox_util.decode_box(
                predictions=outputs,
                anchors=self.anchors,
                image_shape=image_shape,
                input_shape=self.input_shape,
                letterbox_image=self.letterbox_image,
                nms_iou=self.nms_iou,
                confidence=self.confidence
            )

            if len(results[0]) <= 0:
                return image

            top_label = np.array(results[0][:, 4], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(
            font='./model_data/simhei.ttf',
            size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')
        )
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 2)

        for i, c, in enumerate(top_label):
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
                text_origin = np.array([left, top-label_size[1]])
            else:
                text_origin = np.array([left, top+1])

            for j in range(thickness):
                draw.rectangle(
                    xy=[left+j, top+j, right-j, bottom-j],
                    outline=self.colors[c]
                )
            draw.rectangle(
                xy=[tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c]
            )
            draw.text(text_origin, str(label, 'utf-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        images = torch.from_numpy(image_data).type(torch.FloatTensor)
        if self.cuda:
            images = images.cuda()

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                results = self.bbox_util.decode_box(
                    predictions=outputs,
                    anchors=self.anchors,
                    image_shape=image_shape,
                    input_shape=self.input_shape,
                    letterbox_image=self.letterbox_image,
                    nms_iou=self.nms_iou,
                    confidence=self.confidence
                )
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, 'detection-results/' + image_id + '.txt'), 'w')
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            results = self.bbox_util.decode_box(
                predictions=outputs,
                anchors=self.anchors,
                image_shape=image_shape,
                input_shape=self.input_shape,
                letterbox_image=self.letterbox_image,
                nms_iou=self.nms_iou,
                confidence=self.confidence
            )

            if len(results[0]) <= 0:
                return image

            top_label = np.array(results[0][:, 4], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write(
                "%s %s %s %s %s %s\n" %
                (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom)))
            )
        f.close()
        return


if __name__ == '__main__':
    ssd_predictor = SSDPredictor()
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
                r_image = ssd_predictor.detect_image(image)
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
            frame = np.array(ssd_predictor.detect_image(frame))
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
                tact_time = ssd_predictor.get_FPS(image, test_interval)
                print(str(tact_time) + ' second, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # 遍历文件夹进行检测并保存
    elif mode == 'dir_predict':
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.ppm', '.tif', 'tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = ssd_predictor.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

