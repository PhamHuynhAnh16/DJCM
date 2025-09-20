import torch

import torch.nn as nn
import torch.nn.functional as F

def FL(inputs, targets, alpha, gamma, weight_t=None):
    loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    weight = torch.ones(inputs.shape, dtype=torch.float).to(inputs.device)

    weight[targets == 1] = float(alpha)
    loss_w = F.binary_cross_entropy(inputs, targets, weight=weight, reduce=False)

    pt = torch.exp(-loss)
    weight_gamma = (1 - pt) ** gamma

    if weight_t is not None:
        weight_gamma = weight_gamma * weight_t

    F_loss = torch.mean(weight_gamma * loss_w)
    return F_loss

def bce(input, target):
    bce = nn.BCELoss(reduce=False)
    loss = bce(input, target)

    return torch.mean(loss)