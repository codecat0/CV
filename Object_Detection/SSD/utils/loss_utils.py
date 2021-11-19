# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loss_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
import torch.nn as nn


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        super(MultiboxLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception("Only 0 as background label id is supported")
        self.background_label_id = background_label_id
        self.negatives_for_hard = torch.FloatTensor([negatives_for_hard])[0]

    @staticmethod
    def _l1_smmooth_loss(y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    @staticmethod
    def _softmax(y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred), dim=-1)
        return softmax_loss

    def forward(self, y_true, y_pred):
        """
        :param y_true: batch_size, 8732, 4 + num_classes + 1
        :param y_pred: loc: (batch_sizem, 8732, 4); conf: (batch_size, 8732, num_classes)
        """
        num_boxes = y_true.size()[1]
        y_pred = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim=-1)

        conf_loss = self._softmax(y_true[:, :, 4:-1], y_pred[:, :, 4:])
        loc_loss = self._l1_smmooth_loss(y_true[:, :, :4], y_pred[:, :, :4])

        # 获取所有正样本的损失
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1], dim=1)
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1], dim=1)

        # 每张图正样本数量
        num_pos = torch.sum(y_true[:, :, -1], dim=-1)
        # 每张图负样本数量
        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)

        # 查看一个batch中哪些图片是有正样本的
        pos_num_neg_mask = num_pos > 0
        # 如果所有的图，正样本的数量均为0，那么则默认选取100个先验框作为负样本
        has_pos = torch.sum(pos_num_neg_mask)
        num_neg_batch = torch.sum(num_neg) if has_pos > 0 else self.negatives_for_hard

        # 获取难分类样本
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1

        # 把没有包含物体并且将不是背景的概率求和，求和后的概率越大，代表越难分类样本
        max_confs = (1 - y_true[:, :, -1]) * torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)
        max_confs = max_confs.view(-1)

        # 在整个batch里面选取最难分类的num_neg_batch个
        _, indices = torch.topk(max_confs, k=int(num_neg_batch.cpu().numpy()))
        # neg_conf_loss = torch.gather(conf_loss.view(-1), dim=0, index=indices)
        neg_conf_loss = conf_loss.view(-1)[indices]

        # 归一化
        num_pos = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        total_loss = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        total_loss /= torch.sum(num_pos)
        return total_loss


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
