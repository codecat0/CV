# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import FCNDataset, fcn_dataset_collate
from nets.fcn import FCN
from nets.fcn import weights_init
from utils.callbacks import LossHistory
from utils.train_utils import train_one_epoch


def main(args):
    # 是否使用GPU训练
    cuda = False

    # 输入到FCN网络中图像的尺寸
    input_shape = (512, 512)

    # 使用多少线程读取数据集
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    # 类别权重，默认都为1
    cls_weights = np.ones([args.num_classes], np.float32)

    # 模型定义
    model = FCN(
        backbone=args.backbone,
        pretrained=args.pretrained,
        num_classes=args.num_classes
    )

    # 初始化模型权重
    if not args.pretrained:
        weights_init(model)

    # 载入模型权重
    if args.model_path != '':
        print('Load weights {}.'.format(args.model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 多GPU训练
    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    # 读取数据集对应的txt文件，该txt文件存储的是图像名
    with open(os.path.join(args.data_path, 'ImageSets/Segmentation/train.txt'), 'r') as f:
        train_lines = f.readlines()

    with open(os.path.join(args.data_path, 'ImageSets/Segmentation/val.txt'), 'r') as f:
        val_lines = f.readlines()

    # 记录损失等相关日志的变化
    loss_history = LossHistory('logs/')

    # 优化器及学习率变化
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    # 训练集
    train_dataset = FCNDataset(
        annotation_lines=train_lines,
        input_shape=input_shape,
        num_classes=args.num_classes,
        train=True,
        dataset_path=args.data_path
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=fcn_dataset_collate
    )

    # 验证集
    val_dataset = FCNDataset(
        annotation_lines=val_lines,
        input_shape=input_shape,
        num_classes=args.num_classes,
        train=False,
        dataset_path=args.data_path
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=fcn_dataset_collate
    )

    # 是否采用冻结部分权重训练
    if args.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    for epoch in range(args.init_epoch, args.freeze_epoch):
        train_one_epoch(
            model=model,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            Epochs=args.freeze_epoch,
            cuda=cuda,
            dice_loss=args.dice_loss,
            focal_loss=args.focal_loss,
            cls_weights=cls_weights,
            num_classes=args.num_classes
        )
        lr_scheduler.step()

    if args.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = True

    for epoch in range(args.freeze_epoch, args.sum_epoch):
        train_one_epoch(
            model=model,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            Epochs=args.sum_epoch,
            cuda=cuda,
            dice_loss=args.dice_loss,
            focal_loss=args.focal_loss,
            cls_weights=cls_weights,
            num_classes=args.num_classes
        )
        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--backbone', type=str, default='vgg16')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='../../data/VOCdevkit/VOC2007')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--freeze_train', type=bool, default=True)
    parser.add_argument('--init_epoch', type=int, default=0)
    parser.add_argument('--freeze_epoch', type=int, default=50)
    parser.add_argument('--sum_epoch', type=int, default=100)
    parser.add_argument('--dice_loss', type=bool, default=False)
    parser.add_argument('--focal_loss', type=bool, default=False)

    args = parser.parse_args()
    main(args)