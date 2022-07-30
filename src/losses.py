import torch
from torch.nn import functional as F


def dice_loss(pred, true, smooth=1e-3):

    true = (F.one_hot(true.to(torch.int64), num_classes=2)).type(torch.float32)
    pred = F.softmax(pred.permute(0, 2, 3, 1).contiguous(), dim=-1)

    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


def mse_loss(pred, true):

    loss = pred - true
    loss = (loss * loss).mean()
    return loss


def combined_loss(pred, true):

    mse = mse_loss(pred[:, 2, :, :], true[:, :, :, 1])
    dice = dice_loss(pred[:, :2, :, :], true[:, :, :, 0])

    return 1 * (mse) + 2 * (dice)
