# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : predict.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import colorsys
import copy
import time

from tqdm import tqdm
from cv2 import cv2
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.pspnet import PSPNet
from dataset.utils import cvtColor, preprocess_input, resize_image


class PSPNetPredictor(object):
    # 模型权重、类别数、特征提取主干网络、输入图像的尺寸、下采样倍数、识别结果与原图是否混合、是否使用GPU
    _defaults = {
        'model_path': './model_data/pspnet_resnet50.pth',
        'num_classes': 21,
        'backbone': 'resnet50',
        'input_shape': (473, 373),
        'downsample_factor': 16,
        'blend': True,
        'cuda': True
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 每个类别设置不同的颜色
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 载入模型
        self.generate()

    def generate(self):
        """载入模型与权重"""

        self.net = PSPNet(
            num_classes=self.num_classes,
            downsample_factor=self.downsample_factor,
            pretrained=False,
            backbone=self.backbone,
            aux_branch=False
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        # 多GPU
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net.cuda()

    def detect_image(self, image):
        """检测图像"""

        # 将图像转换为RGB格式
        image = cvtColor(image)
        # 识别结果与原图混合时会用到，保留原图
        old_image = copy.deepcopy(image)

        # 获取原始图像的高、宽
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        # 对原始图像进行resize操作，使其满足模型需要的尺寸大小
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 图片传入网络进行预测
            predict = self.net(images)[0]

            # 取出每一个像素点的类别概率
            predict = F.softmax(predict.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 将图像区域截取出来
            predict = predict[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
            ]

            # 将图像resize至原始大小
            predict = cv2.resize(predict, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            # 取出每一个像素点的种类
            predict = predict.argmax(axis=-1)

        # 创建一个新图，根据每一个像素点的种类赋予对应的颜色
        seg_img = np.zeros((np.shape(predict)[0], np.shape(predict)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((predict[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((predict[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((predict[:, :] == c) * (self.colors[c][2])).astype('uint8')

        # 将新图转换为Image格式
        image = Image.fromarray(np.uint8(seg_img))

        # 将预测结果图与原始图混合
        if self.blend:
            image = Image.blend(old_image, image, 0.7)

        return image

    def get_FPS(self, image, test_interval):
        # 将图像转换为RGB格式
        image = cvtColor(image)

        # 对原始图像进行resize操作，使其满足模型需要的尺寸大小
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 图片传入网络进行预测
            predict = self.net(images)[0]

            # 取出每一个像素点的类别概率
            predict = F.softmax(predict.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)

            # 将图像区域截取出来
            predict = predict[
                      int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                      int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
                      ]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                predict = self.net(images)[0]
                predict = F.softmax(predict.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                predict = predict[
                          int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                          int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
                          ]
        t2 = time.time()
        tack_time = (t2 - t1) / test_interval
        return tack_time

    def get_miou_png(self, image):
        # 将图像转换为RGB格式
        image = cvtColor(image)
        # 识别结果与原图混合时会用到，保留原图
        old_image = copy.deepcopy(image)

        # 获取原始图像的高、宽
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        # 对原始图像进行resize操作，使其满足模型需要的尺寸大小
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 图片传入网络进行预测
            predict = self.net(images)[0]

            # 取出每一个像素点的类别概率
            predict = F.softmax(predict.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 将图像区域截取出来
            predict = predict[
                      int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                      int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
                      ]

            # 将图像resize至原始大小
            predict = cv2.resize(predict, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            # 取出每一个像素点的种类
            predict = predict.argmax(axis=-1)

        image = Image.fromarray(np.uint8(predict))
        return image


if __name__ == '__main__':
    pspnet_predictor = PSPNetPredictor()

    # 用于指定测试的模式
    mode = 'predict'

    # video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    video_path = 0
    video_save_path = ''
    video_fps = 25.0

    # 用于指定测量fps的时候，图片检测的次数
    test_interval = 100

    dir_original_path = 'img/'
    dir_save_path = 'img_out/'

    # 单张图片预测
    if mode == 'predict':
        while True:
            img_path = input('Input image filename:')
            try:
                image = Image.open(img_path)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = pspnet_predictor.detect_image(image)
                r_image.save('./img/1_dr.jpg')
                r_image.show()

    # 视频检测
    elif mode == 'video':
        capture = cv2.VideoCapture(video_path)
        if video_save_path != '':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError('未能正确读取摄像头(视频)，请注意是否正确安装摄像头(是否正确填写视频路径)。')

        fps = 0.0
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转换，BGR->RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(pspnet_predictor.detect_image(frame))
            # 格式转换 RGB->BGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, 'fps= %.2f'%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('video', frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != '':
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print('Video Detection Done!')
        capture.release()
        if video_save_path != '':
            print('Save processed video to the path : ' + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    # 测试fps
    elif mode == 'fps':
        img_path = input('Input image filename:')
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
        else:
            tact_time = pspnet_predictor.get_FPS(image, test_interval)
            print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    # 遍历文件夹进行检测并保存
    elif mode == 'dir_predict':
        img_names = os.listdir(dir_original_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_original_path, img_name)
                image = Image.open(image_path)
                r_image = pspnet_predictor.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("请指定正确的模式：'predict', 'video', 'fps', or 'dir_predict'")
