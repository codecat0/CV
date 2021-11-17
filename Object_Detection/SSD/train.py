# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.ssd_dataset import SSDDataset, ssd_dataset_collate
from dataset.data_utils import get_classe
from nets.ssd import SSD300
from utils.anchor_utils import get_anchors
from utils.loss_utils import MultiboxLoss, weights_init
from utils.callbacks import LossHistory
from utils.train_utils import train_one_epoch


if __name__ == '__main__':
    cuda = False
    classes_path = './model_data/voc_classes.txt'
    model_path = './model_data/ssd_weights.pth'
    pretrained = False

    input_shape = (300, 300)
    anchor_sizes = (30, 60, 111, 162, 213, 264, 315)

    init_epoch = 0
    freeze_epoch = 50
    freeze_batch_size = 4
    freeze_lr = 5e-4

    unfreeze_epoch = 100
    unfreeze_batch_size = 2
    unfreeze_lr = 1e-4

    freeze_train = True

    num_workers = 4

    train_annotation_path = './model_data/train.txt'
    val_annotation_path = './model_data/val.txt'

    class_names, num_classes = get_classe(classes_path)
    num_classes += 1
    anchors = get_anchors(
        input_shape=input_shape,
        anchor_size=anchor_sizes
    )

    model = SSD300(
        num_classes=num_classes,
        pretrained=pretrained
    )

    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    criterion = MultiboxLoss(
         num_classes=num_classes,
         neg_pos_ratio=0.3
    )
    loss_history = LossHistory('logs/')

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        batch_size = freeze_batch_size
        lr = freeze_lr
        start_epoch = init_epoch
        end_epoch = freeze_epoch

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = SSDDataset(
            annotation_lines=train_lines,
            input_shape=input_shape,
            anchors=anchors,
            batch_size=batch_size,
            num_classes=num_classes,
            train=True
        )
        val_dataset = SSDDataset(
            annotation_lines=val_lines,
            input_shape=input_shape,
            anchors=anchors,
            batch_size=batch_size,
            num_classes=num_classes,
            train=False
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ssd_dataset_collate
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=ssd_dataset_collate
        )

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('dataset is too small, please expand dataset!!!')

        if freeze_train:
            for param in model.vgg[:28].parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            train_one_epoch(
                model_train=model_train,
                model=model,
                ssd_loss=criterion,
                loss_history=loss_history,
                optimizer=optimizer,
                epoch=epoch,
                epoch_step=epoch_step,
                epoch_step_val=epoch_step_val,
                train_loader=train_loader,
                val_loader=val_loader,
                sum_epoh=end_epoch,
                cuda=cuda
            )
            lr_scheduler.step()