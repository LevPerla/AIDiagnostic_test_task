import torch

def dice_loss(y_real, y_pred):
    y_pred = torch.sigmoid(y_pred)

    num = y_pred.size(2)
    den = torch.sum(2. * y_real * y_pred) / (torch.sum(y_real) + torch.sum(y_pred))
    res = 1. - den / (num**2)
    return res