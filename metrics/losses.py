import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logit_sum = torch.squeeze(logits, dim=1)
        intersection = (logit_sum * targets).sum(dim=(1, 2))
        union = logit_sum.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2. * intersection + 1e-3) / (union + 1e-3)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.9):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        dice_loss_value = self.dice_loss(logits, targets)
        logits = torch.squeeze(logits, dim=1)
        targets = targets.float()
        cross_entropy_loss_value = self.cross_entropy_loss(logits, targets)
        return self.dice_weight * dice_loss_value + (1 - self.dice_weight) * cross_entropy_loss_value


if __name__ == "__main__":

    num_classes = 1
    output = torch.randn(32, num_classes, 128, 128)
    mask = torch.randint(0, num_classes, (32, 128, 128))

    combined_loss = CombinedLoss()
    loss = combined_loss(output, mask)
    print("Output shape: ", output.shape)
    print("Mask shape: ", mask.shape)
    print(f"Loss: {loss.item()}")
