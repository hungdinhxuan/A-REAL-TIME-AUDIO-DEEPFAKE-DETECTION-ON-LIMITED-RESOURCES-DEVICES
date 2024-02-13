import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

    
# def cutmix_data(x, y, alpha=1.0):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size()[0]
#     rand_index = torch.randperm(x.size()[0]).to(x.device)
#     target_a = y
#     target_b = y[rand_index]
#     bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
#     x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
#     # adjust lambda to exactly match pixel ratio
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
#     return x, target_a, target_b, lam, rand_index
def cutmix_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    target_a = y
    target_b = y[rand_index]
    cut_point = np.int32(x.size()[1] * lam)
    x[:, :cut_point] = x[rand_index, :cut_point]
    # adjust lambda to exactly match ratio
    lam = 1 - (cut_point / x.size()[1])
    return x, target_a, target_b, lam, rand_index


def CutMix(net, inputs, targets, criterion_cls, alpha):
    mixed_x, y_a, y_b, lam_mixup, _ = cutmix_data(inputs, targets, alpha=alpha)
    logit = net(mixed_x)
    loss = criterion_cls(logit, y_a) * lam_mixup + criterion_cls(logit, y_b) * (1. - lam_mixup)
    return logit, loss