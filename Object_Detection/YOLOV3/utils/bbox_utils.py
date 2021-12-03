# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : bbox_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
import torch
from torchvision.ops import nms


class DecodeBox(object):
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.box_attrs = 5 + num_classes

    def decode_box(self, inputs):
        """

        :param inputs: 输入的input一共有三个，他们的shape分别是:
                        batch_size, 3, 13, 13, 25 = 5 + num_classes
                        batch_size, 3, 26, 26, 25
                        batch_size, 3, 52, 52, 25
        """
        outputs = []
        device = inputs[0].device
        for i, input in enumerate(inputs):
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            # 此时获得的scaled_anchors大小是相对于特征层的
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            prediction = input.view(batch_size, len(self.anchors_mask[i]), self.box_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # 先验框的中心位置的调整参数
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])

            # 先验框的宽高调整参数
            w = prediction[..., 2]
            h = prediction[..., 3]

            # 获得置信度，是否有物体
            conf = torch.sigmoid(prediction[..., 4])

            # 种类信息
            cls = torch.sigmoid(prediction[..., 5:])

            # 生成网格，先验框中心，网格左上角
            grid_x = torch.linspace(0, input_width-1, input_width).repeat(input_height, 1).repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).float().to(device)
            grid_y = torch.linspace(0, input_height-1, input_height).repeat(input_width, 1).t().repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).float().to(device)

            # 按照网格格式生成先验框的宽高
            anchor_w = torch.FloatTensor(scaled_anchors).index_select(1, torch.tensor([0])).to(device)
            anchor_h = torch.FloatTensor(scaled_anchors).index_select(1, torch.tensor([1])).to(device)
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, input_height * input_width).view(h.shape)

            # 利用预测结果对先验框进行调整
            pred_boxes = torch.FloatTensor(prediction[..., :4].shape).to(device)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # 将输出结果归一化
            scale = torch.FloatTensor([input_width, input_height, input_width, input_height]).to(device)
            output = torch.cat(
                [
                    pred_boxes.view(batch_size, -1, 4) / scale,
                    conf.view(batch_size, -1, 1),
                    cls.view(batch_size, -1, self.num_classes)
                ], dim=-1
            )
            outputs.append(output.data)
        return outputs

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_threshold=0.5, nms_threshold=0.4):
        """
        进行NMS处理
        :param prediction: [batch_size, num_anchors, 5+num_classes]
        :param num_classes: 类别数
        :param input_shape: 输入到网络的图像尺寸
        :param image_shape: 图像原始尺寸
        :param letterbox_image: 在对图像进行resize操作时，是否保存原始图像比例不变
        :param conf_threshold: 置信度阈值
        :param nms_threshold: NMS处理时的阈值
        """
        # 将预测结果有xywh 转换为 xyxy
        box_corner = prediction.new(prediction.shape)
        box_corner[..., 0] = prediction[..., 0] - prediction[..., 2] / 2
        box_corner[..., 1] = prediction[..., 1] - prediction[..., 3] / 2
        box_corner[..., 2] = prediction[..., 0] + prediction[..., 2] / 2
        box_corner[..., 3] = prediction[..., 1] + prediction[..., 3] / 2
        prediction[..., :4] = box_corner[..., :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # 对种类预测部分取max, class_conf: 所属种类概率 (num_anchors, 1); class_pred: 类别(num_anchors, 1)
            class_conf, class_pred = torch.max(image_pred[:, 5:5+num_classes], dim=1, keepdim=True)
            # 利用置信度进行第一轮筛选
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_threshold).squeeze()
            image_pred = image_pred[conf_mask]
            class_pred = class_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            if len(image_pred) == 0:
                continue

            # detections: (num_anchors, 7) 其中7：x1,y1,x2,y2,obj_conf,cls_conf,cls_pred
            detections = torch.cat([image_pred[:, :5], class_conf.float(), class_pred.float()], dim=1)

            # 获得预测结果中包含的所有种类
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                # 对每一个类别进行NMS处理
                detections_by_class = detections[detections[:, -1] == c]
                keep = nms(
                    boxes=detections_by_class[:, :4],
                    scores=detections_by_class[:, 4] * detections_by_class[:, 5],
                    iou_threshold=nms_threshold
                )
                max_detections = detections_by_class[keep]
                output[i] = max_detections if output[i] is None else torch.cat([output[i], max_detections])

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

    @staticmethod
    def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes