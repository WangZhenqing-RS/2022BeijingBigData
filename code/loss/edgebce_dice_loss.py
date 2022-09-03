# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:02:34 2021

@author: DELL
"""

import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


def edgebce_dice_loss(pred, target, edge):
    
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    dice_loss = DiceLoss(mode='binary',
                           from_logits=False)
    # 交叉熵
    bce_loss = nn.BCELoss(reduction='none')
    
    # 边缘加权
    edge_weight = 4.
    loss_bce = bce_loss(pred, target)
    loss_dice = dice_loss(pred, target)
    edge[edge == 0] = 1.
    edge[edge == 255] = edge_weight
    loss_bce *= edge
    
    # OHEM
    loss_bce_,ind = loss_bce.contiguous().view(-1).sort()
    min_value = loss_bce_[int(0.5*loss_bce.numel())]
    loss_bce = loss_bce[loss_bce>=min_value]
    loss_bce = loss_bce.mean()
    loss = loss_bce + loss_dice
    
    return loss


