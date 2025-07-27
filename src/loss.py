import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def mae(input, target, weight=None):
    l1_loss = nn.L1Loss(reduce=False)
    loss = l1_loss(input, target)
    if weight is not None:
        loss = weight * loss
    return torch.mean(loss)


def mse(input, target, weight=None):
    l2_loss = nn.MSELoss(reduce=False)
    loss = l2_loss(input, target)
    if weight is not None:
        loss = weight * loss
    return torch.mean(loss)


def ce(input, target, weight=None):
    ce = nn.CrossEntropyLoss(reduce=False)
    loss = ce(input, target)
    if weight is not None:
        loss = loss * weight
    return torch.mean(loss)


def bce(input, target, weight=None):
    bce = nn.BCELoss(reduce=False)
    loss = bce(input, target)
    if weight is not None:
        loss = loss * weight
    return torch.mean(loss)


def FL(inputs, targets, alpha, gamma, weight_t=None):
    min_len = min(inputs.size(1), targets.size(1))
    inputs = inputs[:, :min_len, :]
    targets = targets[:, :min_len, :]

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


def dynamic_weight_average(loss_t_1, loss_t_2, T=2):
    """

    :param loss_t_1: 每个task上一轮的loss列表，并且为标量
    :param loss_t_2:
    :return:
    """
    # 第1和2轮，w初设化为1，lambda也对应为1
    if not loss_t_1 or not loss_t_2:
        return [1, 1]

    assert len(loss_t_1) == len(loss_t_2)
    task_n = len(loss_t_1)

    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

    lamb = [math.exp(v / T) for v in w]

    lamb_sum = sum(lamb)

    return [task_n * l / lamb_sum for l in lamb]
