import torch
from torch import nn
import torch.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        assert logits.shape == targets.shape, "Logits and targets must have the same shape"

        num_classes = logits.shape[1]
        dice_loss = 0.0

        for c in range(num_classes):
            logit = logits[:, c, :, :]
            target = targets[:, c, :, :].float()
            logit = torch.sigmoid(logit)

            intersection = (logit * target).sum(dim=(1, 2))
            union = logit.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice.mean()

        return dice_loss / num_classes


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        tversky_loss = 0.0
        for c in range(num_classes):
            logit = logits[:, c, :, :]
            target = targets[:, c, :, :].float()
            true_pos = (logit * target).sum(dim=(1, 2))
            false_neg = ((1 - logit) * target).sum(dim=(1, 2))
            false_pos = (logit * (1 - target)).sum(dim=(1, 2))
            tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
            tversky_loss += 1 - tversky.mean()
        return tversky_loss / num_classes


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy_loss(logits, targets.argmax(dim=1), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.9):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()  # TverskyLoss()  #
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
