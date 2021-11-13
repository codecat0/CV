# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from tqdm import tqdm
import torch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, trainer, loss_history, optimizer, epoch, epoch_step, epoch_step_val, train_loader, val_loader, sum_epoch, cuda):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0

    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{sum_epoch}', postfix=dict, miniters=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_step:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()

            rpn_loc, rpn_cls, roi_loc, roi_cls, total = trainer.train_step(images, boxes, labels, 1)
            total_loss += total.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss = roi_loc.item()
            roi_cls_loss += roi_cls.item()

            pbar.set_postfix(
                **{
                    'total_loss': total_loss / (iteration + 1),
                    'rpn_loc_loss': rpn_loc_loss / (iteration + 1),
                    'rpn_cls_loss': rpn_cls_loss / (iteration + 1),
                    'roi_loc_loss': roi_loc_loss / (iteration + 1),
                    'roi_cls_loss': roi_cls_loss / (iteration + 1),
                    'lr': get_lr(optimizer)
                }
            )
            pbar.update(1)

    print('Finish Train')

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1} / {sum_epoch}', postfix=dict, miniters=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= epoch_step_val:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                if cuda:
                    images = images.cuda()

            trainer.optimizer.zero_grad()
            _, _, _, _, val_total = trainer.forward(images, boxes, labels, 1)
            val_loss += val_total.item()
            pbar.set_postfix(
                **{
                    'val_loss': val_loss / (iteration + 1)
                }
            )
            pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(total_loss / epoch_step, val_loss / epoch_step)
    print('Epoch:' + str(epoch+1) + '/' + str(sum_epoch))
    print('Total Loss: %.3f || Val loss: %.3f' % (total_loss / epoch_step, val_loss /epoch_step))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch+1, total_loss/epoch_step, val_loss/epoch_step))


def weight_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and class_name.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise ValueError("initialization method [%s] is not implement" % init_type)
        elif class_name.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)