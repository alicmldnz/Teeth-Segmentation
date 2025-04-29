import torch.nn as nn
import torch
import torch.nn.functional as F

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        ce_loss = self.ce(output, target)

        # Dice Loss hesapla
        pred = torch.softmax(output, dim=1)
        target_onehot = torch.nn.functional.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        intersection = (pred * target_onehot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        total_loss = ce_loss + dice_loss

        return total_loss
