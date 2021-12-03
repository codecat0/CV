# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loss_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import math
import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]):
        super(YOLOLoss, self).__init__()
        # anchors: [[10, 13], [16, 30], [33, 23], -> 52x52
        #           [30, 61],[62, 45],[59, 119],  -> 26x26
        #           [116,90],[156,198],[373,326]] -> 13x13
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.ignore_threshold = 0.5
        self.cuda = cuda

    @staticmethod
    def MSELoss(pred, target):
        return torch.pow(pred - target, 2)

    @staticmethod
    def BCELoss(pred, target):
        epislon = 1e-7
        pred = torch.clip(pred, min=epislon, max=1.0 - epislon)
        return -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)

    def forward(self, l, input, targets=None):
        """
        计算损失
        :param l: 当前输入的数第几个预测特征层，共三层
        :param input: batch_size, 3*(5+num_classes), 13, 13 or
        #             batch_size, 3*(5+num_classes), 26, 26 or
        #             batch_size, 3*(5+num_classes), 52, 52
        :param targets: 真实框,
                        shape: (num_boxes, 5) 其中5: (x, y, w, h, class)
        """
        # 获得图片数量，特征层的高和宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 计算步长
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 此时获得的scaled_anchors大小是相对于特征层的
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors[self.anchors_mask[l]]]

        # batch_size, 3*(5+num_classes), h, w -> batch_size, 3, h, w, 5 + num_classes
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]

        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])

        # 获得类别信息
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获得网络应该有的预测结果
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        # 忽略先验框与真实框IOU较大但又没匹配真实框的先验框
        noobj_mask = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()

        # 真实框越大，比重越小，小框的比重更大
        box_loss_scale = 2 - box_loss_scale

        # 计算中心偏移情况的loss，只有正样本有定位损失
        loss_x = torch.sum(self.MSELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4])
        loss_y = torch.sum(self.MSELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])

        # 计算宽高调整值的loss
        loss_w = torch.sum(self.MSELoss(w, y_true[..., 2]) * box_loss_scale * y_true[..., 4])
        loss_h = torch.sum(self.MSELoss(h, y_true[..., 3]) * box_loss_scale * y_true[..., 4])

        # 计算置信度的loss
        loss_conf = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + torch.sum(
            self.BCELoss(conf, y_true[..., 4]) * noobj_mask)

        # 计算类别损失
        loss_cls = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos

    def get_target(self, l, targets, anchors, in_h, in_w):
        # 计算一共有多少张图片
        bs = len(targets)
        # 用于选取哪些先验框不包含物体，初始时全部标记为不含物体
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # 会给小目标增加权重，让网络更加去关注小目标
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # 真实预测结果
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)

        for b in range(bs):
            if len(targets[b]) == 0:
                continue

            batch_target = torch.zeros_like(targets[b])
            # 计算出正样本在特征层上的中心点
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]

            gt_box_shapes = torch.FloatTensor(batch_target[:, 2:4])

            anchor_shapes = torch.FloatTensor(anchors)

            # 计算IOU，获取每一个真实框IOU值最大的先验框的索引
            best_ns = torch.argmax(self.calculate_iou_wh(gt_box_shapes, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                # 获得真实框属于哪个网格点
                j = torch.floor(batch_target[t, 0]).long()
                i = torch.floor(batch_target[t, 1]).long()
                # 取出真实框的种类
                c = batch_target[t, 4].long()
                # 将此先验框标记为包含物体
                noobj_mask[b, best_n, i, j] = 0
                # 获取先验框调整参数
                y_true[b, best_n, i, j, 0] = batch_target[t, 0] - j.float()
                y_true[b, best_n, i, j, 1] = batch_target[t, 1] - i.float()
                y_true[b, best_n, i, j, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                y_true[b, best_n, i, j, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                y_true[b, best_n, i, j, 4] = 1
                y_true[b, best_n, i, j, c + 5] = 1

                # 大目标loss权重小，小目标loss权重大
                box_loss_scale[b, best_n, i, j] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    @staticmethod
    def calculate_iou_wh(box_a, box_b):
        """
        计算IOU
        :param box_a: 真实框的wh
        :param box_b: 先验框的wh
        :return:
        """
        w_min = torch.min(box_a[:, None, 0], box_b[:, 0])
        h_min = torch.min(box_a[:, None, 1], box_b[:, 1])

        inter = w_min * h_min
        box_a_area = box_a[:, 0] * box_a[:, 1]
        box_b_area = box_b[:, 0] * box_b[:, 1]
        union = box_a_area[:, None] + box_b_area[None]

        return inter / (union - inter)

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        # 计算一共有多少张图片
        bs = len(targets)
        device = x.device
        # 生成网格，先验框中心，网格左上角
        # (in_w) -> (in_h, in_w) -> (bs * 3, in_h, in_w) -> (bs, 3, in_h, in_w)
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * len(self.anchors_mask[l])), 1,
                                                                          1).view(x.shape).float().to(device)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * len(self.anchors_mask[l])), 1,
                                                                          1).view(y.shape).float().to(device)

        # 生成先验框的宽高
        anchor_w = torch.from_numpy(np.array(scaled_anchors)).index_select(1, torch.tensor([0])).long().to(device)
        anchor_h = torch.from_numpy(np.array(scaled_anchors)).index_select(1, torch.tensor([1])).long().to(device)

        # (3) -> (bs, 3) -> (bs, 3*in_h*in_w) -> (bs, 3, in_h, in_w)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes_x = torch.unsqueeze(x.data + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y.data + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])

                #  计算出正样本在特征层上的中心点
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4]

                # 计算交并比, shape: (num_true_box, num_anchors)
                anch_ious = self.calculate_iou_xywh(batch_target, pred_boxes_for_ignore)
                # 每个先验框对应真实框的最大IOU
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                # (num_anchors,) -> (3, in_h, in_w)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask

    @staticmethod
    def calculate_iou_xywh(box_a, box_b):
        """
        计算IOU
        :param box_a: 真实框信息：(num_true_box, 4) 4: x,y,w,h
        :param box_b: 先验框信息：(num_anchors, 4) 4: x,y,w,h
        """
        # 计算真实框的左上角和右下角
        box_a_x1, box_a_x2 = box_a[:, 0] - box_a[:, 2] / 2, box_a[:, 0] + box_a[:, 2] / 2
        box_a_y1, box_a_y2 = box_a[:, 1] - box_a[:, 3] / 2, box_a[:, 1] + box_a[:, 3] / 2
        # 计算先验框获得的预测框的左上角和右下角
        box_b_x1, box_b_x2 = box_b[:, 0] - box_b[:, 2] / 2, box_b[:, 0] + box_b[:, 2] / 2
        box_b_y1, box_b_y2 = box_b[:, 1] - box_b[:, 3] / 2, box_b[:, 1] + box_b[:, 3] / 2

        inter = (torch.min(box_a_x2[:, None], box_b_x2) - torch.max(box_a_x1[:, None], box_b_x1)).clamp(0) * \
                (torch.min(box_a_y2[:, None], box_b_y2) - torch.max(box_a_y1[:, None], box_b_y1)).clamp(0)

        w_a, h_a = box_a_x2 - box_a_x1, box_a_y2 - box_a_y1
        w_b, h_b = box_b_x2 - box_b_x1, box_b_y2 - box_b_y1
        area_a = w_a * h_a
        area_b = w_b * h_b
        union = area_a[:, None] + area_b - inter
        return inter / union

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)