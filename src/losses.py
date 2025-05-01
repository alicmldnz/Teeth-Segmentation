import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss (background hariç), ignore_index destekli
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W], raw model outputs
        targets: [B, H, W], values in 0..C-1 or ignore_index
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Mask out ignore_index for one-hot
        if self.ignore_index is not None:
            # Replace ignore_index with background=0 for one-hot purposes
            targets_for_one_hot = targets.clone()
            targets_for_one_hot[targets_for_one_hot == self.ignore_index] = 0
        else:
            targets_for_one_hot = targets

        # One-hot encode
        with torch.no_grad():
            targets_onehot = F.one_hot(targets_for_one_hot, num_classes)  # [B, H, W, C]
            targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Apply ignore mask
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).float()  # [B,1,H,W]
            probs = probs * mask
            targets_onehot = targets_onehot * mask

        # Compute Dice per class
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # Exclude background class
        dice_score = dice_score[1:]

        # Dice loss = 1 - mean Dice
        loss = 1.0 - dice_score.mean()
        return loss

class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy + Dice Loss
    """
    def __init__(
        self,
        weight: torch.Tensor = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = None
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.ce_w * loss_ce + self.dice_w * loss_dice
