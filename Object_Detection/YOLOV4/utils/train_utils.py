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
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= len(train_loader):
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(target).type(torch.FloatTensor).cuda() for target in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(target).type(torch.FloatTensor) for target in targets]

            optimizer.zero_grad()
            outputs = model(images)

            loss_value_all = 0
            num_pos_all = 0

            for l in range(len(outputs)):
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos

            loss_value = loss_value_all / num_pos_all

            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()

            pbar.set_postfix(
                **{
                    'loss' : loss / (iteration+1),
                    'lr': get_lr(optimizer)
                }
            )
            pbar.update(1)
    print('Finish Train')

    model.eval()
    print('Start Validation')
    with tqdm(total=len(val_loader), desc=f'Epoch {epoch+1}/{Epochs}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= len(val_loader):
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(target).type(torch.FloatTensor).cuda() for target in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(target).type(torch.FloatTensor) for target in targets]

            optimizer.zero_grad()
            outputs = model(images)

            loss_value_all = 0
            num_pos_all = 0

            for l in range(len(outputs)):
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos

            loss_value = loss_value_all / num_pos_all

            val_loss += loss_value.item()
            pbar.set_postfix(
                **{
                    'val_loss': val_loss / (iteration+1)
                }
            )
            pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(loss / len(train_loader), val_loss / len(val_loader))
    print('Epoch:' + str(epoch+1) + '/' + str(Epochs))
    print('Total Loss: %.3f || Val Loss: %.3f' % (loss / len(train_loader), val_loss / len(val_loader)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%3.f.pth' % (epoch+1, loss/len(train_loader), val_loss/len(val_loader)))