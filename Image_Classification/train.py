# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import math
import argparse
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import get_dataset_dataloader
from utils.train_val_utils import train_one_epoch, evaluate
from models.base_model import BaseModel


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter('./logs')

    # 获取数据集
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataset_dataloader(args.data_path, args.batch_size)

    # 获取模型
    model = BaseModel(name=args.model_name, num_classes=args.num_classes).to(device)

    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5E-5)

    # cosine
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0

    start = time.time()
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch
        )

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch
        )

        # tensorboard
        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/" + args.model_name + ".pth")
    end = time.time()
    print("Training 耗时为:{:.1f}".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default='./data/flower_photos')
    parser.add_argument('--flag', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')

    opt = parser.parse_args()
    print(opt)
    main(opt)