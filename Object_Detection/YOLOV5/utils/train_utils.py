# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import math
from functools import partial
import torch
from tqdm import tqdm


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            out_lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2)
        elif iters >= total_iters - no_aug_iter:
            out_lr = min_lr
        else:
            out_lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return out_lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError('step_size must above 1.')
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == 'cos':
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        deacy_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, deacy_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

            for l in range(len(outputs)):
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