# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : callbacks.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import datetime
import scipy.signal
import matplotlib.pyplot as plt


class LossHistory(object):
    def __init__(self, log_dir):
        cur_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(cur_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, 'loss_'+str(self.time_str))
        self.losses = []
        self.val_losses = []

        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_losses.append(val_loss)

        with open(os.path.join(self.save_path, 'epoch_loss_'+str(self.time_str)+'.txt'), 'a') as f:
            f.write(str(loss))
            f.write('\n')

        with open(os.path.join(self.save_path, 'epoch_val_loss_'+str(self.time_str)+'.txt'), 'a') as f:
            f.write(str(val_loss))
            f.write('\n')

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), 'blue', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'epoch_loss_'+str(self.time_str)+'.png'), dpi=600)

        plt.cla()
        plt.close('all')
