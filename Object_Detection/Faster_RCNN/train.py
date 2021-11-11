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

from nets.faster_rcnn import FasterRCNN
from nets.frcnn_training import FasterRCNNTrainer
from utils.callbacks import LossHistory
from dataset.faster_rcnn_dataset import FRCNNDataset, frcnn_dataset_collate
from utils.data_utils import get_classes
from utils.train_utils import train_one_epoch, weight_init


def main(args):
    cuda = torch.cuda.is_available()

    input_shape = (600, 600)
    anchor_scales = (8, 16, 32)

    # 获取类别和类别数
    class_names, num_classes = get_classes(args.classes_path)
    # 定义模型
    model = FasterRCNN(
        num_classes=num_classes,
        anchor_scales=anchor_scales,
        backbone=args.backbone,
        pretrained=args.pretrained
    )

    if args.pretrained:
        weight_init(model)

    if args.model_path != '':
        # 载入权重
        print('Load weights {}.'.format(args.model_path))
        model_dict = model.state_dict()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(args.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if cuda:
        # 多GPU训练（这里没有设置GPU卡数）
        model_train = torch.nn.DataParallel(model)
        # 加速网络的训练
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory('logs/')

    # 读取数据集对应的txt
    with open(args.train_annotation_path) as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 第一阶段训练 [init_epoch, freeze_epoch]
    if args.freeze_train:
        batch_size = args.freeze_batch_size
        lr = args.freeze_lr
        # 冻结特征提取网络
        for param in model.extractor.parameters():
            param.requires_grad = False
    else:
        batch_size = args.unfreeze_batch_size
        lr = args.unfreeze_lr

    start_epoch = args.init_epoch
    end_epoch = args.freeze_epoch

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    train_dataset = FRCNNDataset(
        annotation_lines=train_lines,
        input_shape=input_shape,
        train=True
    )
    val_dataset = FRCNNDataset(
        annotation_lines=val_lines,
        input_shape=input_shape,
        train=False
    )
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        collate_fn=frcnn_dataset_collate
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        collate_fn=frcnn_dataset_collate
    )

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('dataset is too small, can not train, please expand dataset')

    if batch_size < 4:
        model.freeze_bn()

    trainer = FasterRCNNTrainer(model, optimizer)

    for epoch in range(start_epoch, end_epoch):
        train_one_epoch(
            model=model,
            trainer=trainer,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            epoch_step=epoch_step,
            epoch_step_val=epoch_step_val,
            train_loader=train_loader,
            val_loader=val_loader,
            sum_epoch=end_epoch,
            cuda=cuda
        )
        lr_scheduler.step()

    # 第二阶段训练 [freeze_epoch, sum_epoch]
    if args.freeze_train:
        # 解冻特征提取网络
        for param in model.extractor.parameters():
            param.requires_grad = True

    batch_size = args.unfreeze_batch_size
    lr = args.unfreeze_lr

    start_epoch = args.freeze_epoch
    end_epoch = args.sum_epoch

    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    train_dataset = FRCNNDataset(
        annotation_lines=train_lines,
        input_shape=input_shape,
        train=True
    )
    val_dataset = FRCNNDataset(
        annotation_lines=val_lines,
        input_shape=input_shape,
        train=False
    )
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        collate_fn=frcnn_dataset_collate
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        collate_fn=frcnn_dataset_collate
    )

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('dataset is too small, can not train, please expand dataset')

    if batch_size < 4:
        model.freeze_bn()

    trainer = FasterRCNNTrainer(model, optimizer)

    for epoch in range(start_epoch, end_epoch):
        train_one_epoch(
            model=model,
            trainer=trainer,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            epoch_step=epoch_step,
            epoch_step_val=epoch_step_val,
            train_loader=train_loader,
            val_loader=val_loader,
            sum_epoch=end_epoch,
            cuda=cuda
        )
        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes_path', type=str, default='./model_data/voc_classes.txt')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--model_path', type=str, default='./model_data/voc_weights_resnet.pth')
    parser.add_argument('--train_annotation_path', type=str, default='./model_data/train.txt')
    parser.add_argument('--val_annotation_path', type=str, default='./model_data/val.txt')
    parser.add_argument('--freeze_train', action='store_true')
    parser.add_argument('--freeze_batch_size', type=int, default=4)
    parser.add_argument('--freeze_lr', type=float, default=1e-4)
    parser.add_argument('--unfreeze_batch_size', type=int, default=2)
    parser.add_argument('--unfreeze_lr', type=float, default=1e-5)
    parser.add_argument('--init_epoch', type=int, default=0)
    parser.add_argument('--freeze_epoch', type=int, default=50)
    parser.add_argument('--sum_epoch', type=int, default=100)

    args = parser.parse_args()
    main(args)
