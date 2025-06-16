import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def tce_loss(y, t, drop_rate, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(y, t, reduction="none")

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update], pos_weight=pos_weight)

    return loss_update


def rce_loss(y, t, alpha=0.2, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(y, t, reduction="none", pos_weight=pos_weight)
    y_ = torch.sigmoid(y).detach()
    weight = torch.pow(y_, alpha) * t + torch.pow((1-y_), alpha) * (1-t)
    loss_ = loss * weight
    loss_ = torch.mean(loss_)
    return loss_


class TCE_Loss(nn.Module):
    def __init__(self, num_iterations, drop_rate=0.2, exponent=1):
        super().__init__()
        self.num_iterations = num_iterations
        self.drop_rate = drop_rate
        self.exponent = exponent
        self.drop_rate_ls = np.linspace(0, self.drop_rate**self.exponent, self.num_iterations)

    def forward(self, y, t, n_iterations, pos_weight=None):
        drop_rate = self.drop_rate_schedule(n_iterations)
        return tce_loss(y, t, drop_rate, pos_weight)


    def drop_rate_schedule(self, iteration):

        if iteration < self.num_iterations:
            return self.drop_rate_ls[iteration]
        else:
            return self.drop_rate


class RCE_Loss(nn.Module):
    def __init__(self, num_iterations, drop_rate=0.2, exponent=1):
        super().__init__()
        self.num_iterations = num_iterations
        self.drop_rate = drop_rate
        self.exponent = exponent
        self.drop_rate_ls = np.linspace(0, self.drop_rate**self.exponent, self.num_iterations)

    def forward(self, y, t, n_iterations, pos_weight=None):
        drop_rate = self.drop_rate_schedule(n_iterations)
        return tce_loss(y, t, drop_rate, pos_weight)


    def drop_rate_schedule(self, iteration):

        if iteration < self.num_iterations:
            return self.drop_rate_ls[iteration]
        else:
            return self.drop_rate
