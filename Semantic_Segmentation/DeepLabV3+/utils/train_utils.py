# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
# !/usr/bin/env python
# -*-coding:utf-8 -*-
import math
from functools import partial
import torch
from tqdm import tqdm
from .loss_utils import CE_loss, Focal_loss, Dice_loss
from .metrics_utils import f_score


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            out_lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
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
        warmup_total_iters = min(max(warup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, loss_history, optimizer, epoch, train_loader, val_loader, Epochs, cuda, dice_loss, focal_loss, cls_weights, num_classes):
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    model.train()
    print('Start Train')
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1} / {Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= len(train_loader):
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model(imgs)
            if focal_loss:
                loss = Focal_loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice_loss = Dice_loss(outputs, labels)
                loss = loss + main_dice_loss

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)
    print('Finish Train')

    model.eval()
    print('Start Validation')
    with tqdm(total=len(val_loader), desc=f'Epoch {epoch+1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= len(val_loader):
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()


                outputs = model(imgs)
                if focal_loss:
                    loss = Focal_loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice_loss = Dice_loss(outputs, labels)
                    loss = loss + main_dice_loss

                _f_score = f_score(outputs, labels)

                val_loss += loss.item()
                val_f_score += _f_score.item()

            pbar.set_postfix(**{
                'total_loss': val_loss / (iteration + 1),
                'f_score': val_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    loss_history.append_loss(epoch+1, total_loss / len(train_loader), val_loss / len(val_loader))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epochs))
    print('Total Loss %.3f || Val Loss %.3f' % (total_loss / len(train_loader), val_loss / len(val_loader)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1), total_loss / len(train_loader), val_loss / len(val_loader)))