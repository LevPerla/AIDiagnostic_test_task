import torch

def dice_coeff(y_real, y_pred):
    smooth = 1e-20
    y_pred = torch.sigmoid(y_pred)

    intersection = (y_pred * y_real).sum()
    res = (2. * intersection + smooth) / (torch.sum(y_real) + torch.sum(y_pred) + smooth)
    return res

def dice_loss(y_real, y_pred):
    return 1. - dice_coeff(y_real, y_pred)