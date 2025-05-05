import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.3),
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.GaussNoise(p=0.2),
        # Cutout, küçük delikler açarak kenarları vurgular
        A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
        # Normalize ve Tensor’a çevir
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def apply_augmentation(image, mask, aug):
    """ Albumentations augmentasyonunu uygular, sonra normalize eder. """
    data = aug(image=image, mask=mask)
    aug_image = data["image"]
    aug_mask  = data["mask"]

    # aug_image ya np.ndarray ya da torch.Tensor olabilir:
    if isinstance(aug_image, np.ndarray):
        aug_image = aug_image.astype(np.float32) / 255.0
    else:  # torch.Tensor
        aug_image = aug_image.float() / 255.0

    # aug_mask genellikle Tensor (Long) oluyor, ama eminsen np.ndarray kontrolü de yapabilirsin:
    if isinstance(aug_mask, np.ndarray):
        aug_mask = torch.from_numpy(aug_mask).long()
    else:
        aug_mask = aug_mask.long()

    return aug_image, aug_mask


