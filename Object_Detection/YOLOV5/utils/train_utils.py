# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import torch
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, yolo_loss, loss_history, optimizer, epoch, train_loader, val_loader, Epochs, cuda):
    loss = 0
    val_loss = 0

    model.train()
    print('Start Train')
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration > len(train_loader):
                break

            images, targets, y_trues = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()
                    targets = [target.cuda() for target in targets]
                    y_trues = [y_true.cuda() for y_true in y_trues]

            optimizer.zero_grad()
            outputs = model(images)

            loss_value_all = 0

            for l in range(outputs):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all += loss_item

            loss_value_all.backward()
            optimizer.step()

            loss += loss_value_all.item()
            pbar.set_postfix(
                **{
                    'loss': loss / (iteration + 1),
                    'lr': get_lr(optimizer)
                }
            )
            pbar.update(1)

    print('Finish Train')

    model.eval()
    print('Start Validation')
    with tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= len(val_loader):
                break

            images, targets, y_trues = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()
                    targets = [target.cuda() for target in targets]
                    y_trues = [y_true.cuda() for y_true in y_trues]

            optimizer.zero_grad()
            outputs = model(images)
            loss_value_all = 0

            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all += loss_item

            val_loss += loss_value_all.item()
            pbar.set_postfix(
                **{
                    'val_loss': val_loss / (iteration + 1)
                }
            )
            pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(epoch+1, loss / len(train_loader), val_loss / len(val_loader))
    print('Epoch:' + str(epoch+1) + '/' + str(Epochs))
    print('Total loss: %.3f || Val loss: %.3f' % (loss / len(train_loader), val_loss / len(val_loader)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch+1, loss/len(train_loader), val_loss/len(val_loader)))