import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.squeeze(dim=1)
        logits = torch.sigmoid(logits)
        targets = torch.sigmoid(targets)
        intersection = (logits * targets).sum(dim=(1, 2))
        union = logits.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, cross_entropy_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight

    def forward(self, logits, targets):
        dice_loss_value = self.dice_loss(logits, targets)
        logits_flat = logits.view(-1, 1, 128, 128)
        targets_flat = targets.view(-1, 128, 128).long()
        cross_entropy_loss_value = self.cross_entropy_loss(logits_flat, targets_flat)
        return self.dice_weight * dice_loss_value + self.cross_entropy_weight * cross_entropy_loss_value


if __name__ == "__main__":

    num_classes = 1
    output = torch.randn(32, num_classes, 128, 128)
    mask = torch.randint(0, num_classes, (32, 128, 128))

    combined_loss = CombinedLoss()
    loss = combined_loss(output, mask)
    print("Output shape: ", output.shape)
    print("Mask shape: ", mask.shape)
    print(f"Loss: {loss.item()}")
