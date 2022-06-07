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

from nets.yolo import YoloBody
from dataset.dataset import YoloDataset, yolo_dataset_collate
from dataset.data_utils import get_classes, get_anchors
from utils.loss_utils import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.train_utils import train_one_epoch, get_lr_scheduler, set_optimizer_lr


def main(args):
    cuda = False
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = (640, 640)
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    # 获取类别名称和anchor
    class_names, num_classes = get_classes(args.classes_path)
    anchors, num_anchors = get_anchors(args.anchors_path)

    # 定义模型
    model = YoloBody(
        anchors_mask=anchors_mask,
        num_classes=num_classes,
        phi=args.phi,
        backbone=args.backbone,
        input_shape=input_shape
    )
    # 模型初始化
    weights_init(model)

    # 加载预训练权重
    if args.model_path != '':
        print('Load weights {}'.format(args.model_path))
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

    # 定义损失函数
    yolo_loss = YOLOLoss(
        anchors=anchors,
        num_classes=num_classes,
        input_shape=input_shape,
        cuda=cuda,
        anchors_mask=anchors_mask,
        label_smoothing=args.label_smoothing
    )

    # 记录loss
    loss_history = LossHistory(log_dir=args.log_dir, model=model, input_shape=input_shape)

    # 读取数据集对应的txt
    with open(args.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    # 优化器及学习率的变化
    init_lr = args.lr
    min_lr = init_lr * 1e-2
    optimizer = optim.Adam(model.parameters(), init_lr)
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type=args.lr_decay_type,
        lr=init_lr,
        min_lr=min_lr,
        total_iters=args.sum_epoch
    )

    # 训练集
    train_dataset = YoloDataset(
        annotation_lines=train_lines,
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        anchors_mask=anchors_mask,
        mosaic=True,
        train=True
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate
    )

    # 验证集
    val_dataset = YoloDataset(
        annotation_lines=val_lines,
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        anchors_mask=anchors_mask,
        mosaic=True,
        train=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate
    )

    for epoch in range(args.sum_epoch):
        set_optimizer_lr(
            optimizer=optimizer,
            lr_scheduler_func=lr_scheduler_func,
            epoch=epoch
        )
        train_one_epoch(
            model=model,
            yolo_loss=yolo_loss,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            Epochs=args.sum_epoch,
            cuda=cuda
        )

    loss_history.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--classes_path', type=str, default='./model_data/voc_classes.txt')
    parser.add_argument('--anchors_path', type=str, default='./model_data/yolo_anchors.txt')
    parser.add_argument('--phi', choices=['s', 'm', 'l', 'x'], default='l')
    parser.add_argument('--backbone', type=str, default='cspdarknet')
    parser.add_argument('--model_path', type=str, default='./model_data/yolov5_l.pth')
    parser.add_argument('--label_smoothing', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--train_annotation_path', type=str, default='./model_data/train.txt')
    parser.add_argument('--val_annotation_path', type=str, default='./model_data/val.txt')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_decay_type', choices=['cos', 'step'], default='step')
    parser.add_argument('--sum_epoch', type=int, default=100)

    args = parser.parse_args()
    main(args)