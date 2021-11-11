# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : frcnn_training.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_utils import bbox_iou, bbox2loc


class AnchorTargetCreator(object):
    def __init__(
            self,
            n_sample=256,
            pos_iou_thresh=0.7,
            neg_iou_thresh=0.3,
            pos_ratio=0.5
    ):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    @staticmethod
    def _calc_ious(anchor, gtbox):
        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        ious = bbox_iou(anchor, gtbox)

        if len(gtbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(gtbox))
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        argmax_ious = ious.argmax(axis=1)
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        max_ious = np.max(ious, axis=1)
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        gt_argmax_ious = ious.argmax(axis=0)
        #   保证每一个真实框都存在对应的先验框
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1

        #   判断正样本数量是否大于128，如果大于则限制在128
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        #   平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(
            self,
            n_sample=128,
            pos_ratio=0.5,
            pos_iou_thresh=0.5,
            neg_iou_thresh_high=0.5,
            neg_iou_thresh_low=0
    ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        # 将gtbox也算为roi
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        #   计算建议框和真实框的重合程度
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            #   获得每一个建议框最对应的真实框  [num_roi, ]
            gt_assignment = iou.argmax(axis=1)
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            max_iou = iou.max(axis=1)
            #   真实框的标签要+1因为有背景的存在
            gt_roi_label = label[gt_assignment] + 1

        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    @staticmethod
    def _faster_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
        """
        回归损失计算
        """
        pred_loc = pred_loc[gt_label>0]
        gt_loc = gt_loc[gt_label>0]

        sigma_squared = sigma ** 2
        regresion_diff = gt_loc - pred_loc
        regresion_loss = torch.where(
            regresion_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regresion_diff ** 2,
            regresion_diff - 0.5 / sigma_squared
        )

        regresion_loss = regresion_loss.sum()
        num_pos = (gt_label>0).sum().float()
        regresion_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regresion_loss

    def forward(self, imgs, bboxes, labels, scale):
        # 获取batch_size以及图像的尺寸
        n = imgs.shape[0]
        img_size = imgs.shape[2:]

        # 获取公用特征层
        base_feature = self.faster_rcnn.extractor(imgs)

        # 利用rpn网络获得调整参数、得分、建议框、先验框
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(base_feature, img_size, scale)

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        for i in range(n):
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[roi_indices==i]
            feature = base_feature[i]

            # 利用真实框和先验框获得建议框网络应该有的预测结果，给每个先验框都打上标签
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor)
            gt_rpn_loc = torch.as_tensor(gt_rpn_loc, device=rpn_loc.device)
            gt_rpn_label = torch.as_tensor(gt_rpn_label, device=rpn_loc.device).long()

            # 分别计算建议框网络的回归损失和分类损失
            rpn_loc_loss = self._faster_rcnn_loc_loss(
                pred_loc=rpn_loc,
                gt_loc=gt_rpn_loc,
                gt_label=gt_rpn_label,
                sigma=self.rpn_sigma
            )

            rpn_cls_loss = F.cross_entropy(
                input=rpn_score,
                target=gt_rpn_label,
                ignore_index=-1
            )

            # 利用真实框和建议框获得RoIHead网络应该有的预测结果
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi=roi,
                bbox=bbox,
                label=label,
                loc_normalize_std=self.loc_normalize_std
            )
            sample_roi = torch.as_tensor(sample_roi, device=feature.device)
            gt_roi_loc = torch.as_tensor(gt_roi_loc, device=feature.device)
            gt_roi_label = torch.as_tensor(gt_roi_label, device=feature.device).long()
            sample_roi_index = torch.zeros(len(sample_roi), device=feature.device)

            roi_cls_loc, roi_score = self.faster_rcnn.head(
                torch.unsqueeze(feature, 0),
                sample_roi,
                sample_roi_index,
                img_size
            )

            # 根据建议框的种类，取出对应的回归预测结果
            n_sample = roi_cls_loc.size()[1]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # 分别计算RoIHead网络的回归损失和分类损失
            roi_loc_loss = self._faster_rcnn_loc_loss(
                pred_loc=roi_loc,
                gt_loc=gt_roi_loc,
                gt_label=gt_roi_label.data,
                sigma=self.roi_sigma
            )
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score[0], gt_roi_label)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        losses = losses + [sum(losses)]
        return losses

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses[-1].backward()
        self.optimizer.step()
        return losses