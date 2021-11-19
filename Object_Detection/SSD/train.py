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

from dataset.ssd_dataset import SSDDataset, ssd_dataset_collate
from dataset.data_utils import get_classes
from nets.ssd import SSD300
from utils.anchor_utils import get_anchors
from utils.loss_utils import MultiboxLoss, weights_init
from utils.callbacks import LossHistory
from utils.train_utils import train_one_epoch


def main(args):
    cuda = False
    input_shape = (300, 300)
    anchor_sizes = (30, 60, 111, 162, 213, 264, 315)
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    class_names, num_classes = get_classes(args.classes_path)
    num_classes += 1
    anchors = get_anchors(
        input_shape=input_shape,
        anchor_size=anchor_sizes
    )

    model = SSD300(
        num_classes=num_classes,
        pretrained=args.pretrained
    )

    if not args.pretrained:
        weights_init(model)
    if args.model_path != '':
        print('Load weights {}.'.format(args.model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    criterion = MultiboxLoss(
         num_classes=num_classes,
         neg_pos_ratio=0.3
    )
    loss_history = LossHistory('logs/')

    with open(args.train_annotation_path) as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path) as f:
        val_lines = f.readlines()

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    train_dataset = SSDDataset(
        annotation_lines=train_lines,
        input_shape=input_shape,
        anchors=anchors,
        num_classes=num_classes,
        train=True
    )
    val_dataset = SSDDataset(
        annotation_lines=val_lines,
        input_shape=input_shape,
        anchors=anchors,
        num_classes=num_classes,
        train=False
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ssd_dataset_collate
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ssd_dataset_collate
    )

    if args.freeze_train:
        for param in model.vgg[:28].parameters():
            param.requires_grad = False

    for epoch in range(args.init_epoch, args.freeze_epoch):
        train_one_epoch(
            model=model,
            ssd_loss=criterion,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            sum_epoh=args.freeze_epoch,
            cuda=cuda
        )
        lr_scheduler.step()

    if args.freeze_train:
        for param in model.vgg[:28].parameters():
            param.requires_grad = True

    for epoch in range(args.freeze_epoch, args.sum_epoch):
        train_one_epoch(
            model=model,
            ssd_loss=criterion,
            loss_history=loss_history,
            optimizer=optimizer,
            epoch=epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            sum_epoh=args.sum_epoch,
            cuda=cuda
        )
        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--classes_path', type=str, default='./model_data/voc_classes.txt')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./model_data/ssd_weights.pth')
    parser.add_argument('--train_annotation_path', type=str, default='./model_data/train.txt')
    parser.add_argument('--val_annotation_path', type=str, default='./model_data/val.txt')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--freeze_train', type=bool, default=False)
    parser.add_argument('--init_epoch', type=int, default=0)
    parser.add_argument('--freeze_epoch', type=int, default=50)
    parser.add_argument('--sum_epoch', type=int, default=100)

    args = parser.parse_args()
    main(args)