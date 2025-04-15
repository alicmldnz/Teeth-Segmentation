import torch
import torch.nn.functional as F

def dice_score(pred, target, num_classes, epsilon=1e-6):
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    dice_total = 0.0

    for cls in range(1, num_classes):  # background hariç
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_total += dice

    return dice_total / (num_classes - 1)
