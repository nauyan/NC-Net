import torch
from torch.nn import functional as F
import numpy as np


def dice_score(pred, true):

    pred = pred[:, :2, :, :]
    true = true[:, :, :, 0]

    pred = F.softmax(pred.permute(0, 2, 3, 1).contiguous(), dim=-1)
    pred = torch.argmax(pred, dim=-1, keepdim=False)

    true = np.copy(true.cpu().detach().numpy())
    pred = np.copy(pred.cpu().detach().numpy())

    inter = true * pred
    denom = true + pred
    return (2.0 * np.sum(inter)) / np.sum(denom)


def mse_metric(pred, true):

    pred = pred[:, 2, :, :]
    true = true[:, :, :, 1]

    true = np.copy(true.cpu().detach().numpy())
    pred = np.copy(pred.cpu().detach().numpy())

    loss = pred - true
    loss = (loss * loss).mean()
    return loss
