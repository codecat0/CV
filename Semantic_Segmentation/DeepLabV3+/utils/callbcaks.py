# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : callbcaks.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import datetime
import os

import scipy.signal as signal
from matplotlib import pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter


class LossHistory(object):
    def __init__(self, log_dir, model, input_shape):
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, 'loss_' + str(self.time_str))
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model=model, input_to_model=dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, 'epoch_loss_' + str(self.time_str) + '.txt'), 'a') as f:
            f.write(str(loss))
            f.write('\n')

        with open(os.path.join(self.log_dir, 'epoch_val_loss_' + str(self.time_str) + '.txt'), 'a') as f:
            f.write(str(val_loss))
            f.write('\n')

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, signal.savgol_filter(self.losses, num, 3), linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, signal.savgol_filter(self.val_loss, num, 3), linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig(os.path.join(self.log_dir, 'epoch_loss_' + str(self.time_str) + '.png'))

        plt.cla()
        plt.close('all')