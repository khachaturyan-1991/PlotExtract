import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (logits * targets_one_hot).sum(dim=(2, 3))
        union = logits.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int = 3, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        targets = targets.long()
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.alpha * ce + (1 - self.alpha) * dice


if __name__ == "__main__":

    num_classes = 3
    output = torch.randn(32, num_classes, 128, 128)
    mask = torch.randint(0, num_classes, (32, 128, 128))

    combined_loss = CombinedLoss()
    loss = combined_loss(output, mask)
    print("Output shape: ", output.shape)
    print("Mask shape: ", mask.shape)
    print(f"Loss: {loss.item()}")
