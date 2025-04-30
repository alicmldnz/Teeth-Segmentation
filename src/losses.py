import torch.nn as nn
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class CombinedLoss(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        ce_loss = self.ce(output, target)

        # Dice Loss hesapla
        pred = torch.softmax(output, dim=1)  # [B, C, H, W]

        # target'ı one-hot encode et
        target_onehot = F.one_hot(target, num_classes=self.num_classes)  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()         # [B, C, H, W]

        # Dice loss'u her sınıf için hesapla
        intersection = (pred * target_onehot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

        dice_loss_per_class = 1 - (2 * intersection + 1e-6) / (union + 1e-6)  # [B, C]
        
        # Background sınıfı (0) hariç ortalama dice al
        dice_loss = dice_loss_per_class[:, 1:].mean()

        total_loss = ce_loss + dice_loss

        return total_loss

