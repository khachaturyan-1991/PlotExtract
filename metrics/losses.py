import torch
from torch import nn


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


class SMSE(nn.Module):

    def __init__(self):
        super(SMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        mse_loss = 0
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]
        for batch in range(batch_size):
            for channel in range(num_classes):
                targets_ch = targets[batch, channel, :, :]
                logits_ch = logits[batch, channel, :, :]
                mask_pos = targets_ch > 0
                if torch.any(mask_pos):
                    mse_loss += self.mse(logits_ch[mask_pos], targets_ch[mask_pos])
                mask_neg = targets_ch == 0
                if torch.any(mask_neg):
                    mse_loss += self.mse(logits_ch[mask_neg], targets_ch[mask_neg])
        return mse_loss / num_classes / batch_size


class CombinedLoss(nn.Module):

    def __init__(self, dice_weight=0.8):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.mse = SMSE()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        dice_loss_value = self.dice_loss(logits, targets)
        mse_loss = self.mse(logits, targets)
        return self.dice_weight * dice_loss_value + (1 - self.dice_weight) * mse_loss


if __name__ == "__main__":

    num_classes = 2
    output = torch.randn([32, num_classes, 128, 128])
    mask = torch.randn([32, num_classes, 128, 128])

    combined_loss = CombinedLoss()
    loss = combined_loss(output, mask)
    print("Output shape: ", output.shape)
    print("Mask shape: ", mask.shape)
    print(f"Loss: {loss.item()}")
