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


def train_one_epoch(model, ssd_loss, loss_history, optimizer, epoch,
                    train_loader, val_loader, sum_epoh, cuda):
    train_loss = 0
    val_loss = 0
    model.train()
    print('Start Train')
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{sum_epoh}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= len(train_loader):
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = torch.from_numpy(targets).type(torch.FloatTensor)

            out = model(images)
            optimizer.zero_grad()
            loss = ssd_loss(targets, out)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(
                **{
                    'train_loss' : train_loss / (iteration+1),
                    'lr' : get_lr(optimizer)
                }
            )
            pbar.update(1)
    print('Finish Train')

    model.eval()
    print('Start Validation')
    with tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}/{sum_epoh}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= len(val_loader):
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = torch.from_numpy(targets).type(torch.FloatTensor)

                out = model(images)
                optimizer.zero_grad()
                loss = ssd_loss.forward(targets, out)
                val_loss += loss.item()

                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(train_loss / len(train_loader), val_loss / len(val_loader))
    print('Epoch:' + str(epoch + 1) + '/' + str(sum_epoh))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss / len(train_loader), val_loss / len(val_loader)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_loss / len(train_loader), val_loss / len(val_loader)))

