# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_val_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import sys

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()

    # 累计损失，累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    dataloader = tqdm(dataloader)
    for step, data in enumerate(dataloader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.requires_grad_(True)
        loss.backward()
        accu_loss += loss.detach()

        dataloader.desc = "[train epoch {}] loss: {:3f}, acc: {:3f}".format(
            epoch, accu_loss.item()/(step+1), accu_num.item()/(step+1)
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item()/len(dataloader), accu_num.item()/sample_num


@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()

    loss_function = nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    dataloader = tqdm(dataloader)
    for step, data in enumerate(dataloader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss.detach()

        dataloader.desc = "[valid epoch {}] loss: {:.3f}, acc {:.3f}".format(
            epoch, accu_loss.item()/(step+1), accu_num.item()/sample_num
        )

    return accu_loss.item()/len(dataloader), accu_num.item()/sample_num