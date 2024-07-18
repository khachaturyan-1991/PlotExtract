import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        intersection = (logits * targets).sum()
        dice = (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        return 1 - dice


def cross_entropy_loss(logits, targets):
    return F.cross_entropy(logits, targets)


class CrossDice(nn.Module):
    def __init__(self, smooth=1, alpha=0.5):
        super(CrossDice, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.alpha = alpha

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        ce = cross_entropy_loss(logits, targets)
        combined = self.alpha * ce + (1 - self.alpha) * dice
        return torch.mean(combined)
