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
from .loss_utils import CE_Loss, Focal_Loss, Dice_Loss
from .metrics_utils import f_score


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
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice_loss = Dice_Loss(outputs, labels)
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
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice_loss = Dice_Loss(outputs, labels)
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

    loss_history.append_loss(total_loss / len(train_loader), val_loss / len(val_loader))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epochs))
    print('Total Loss %.3f || Val Loss %.3f' % (total_loss / len(train_loader), val_loss / len(val_loader)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1), total_loss / len(train_loader), val_loss / len(val_loader)))