import albumentations as A
import numpy as np
import torch

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
    ])

def apply_augmentation(image, mask, transform):
    # 🚀 Albumentations uint8 input bekler → Convert et
    image_uint8 = (image * 255).astype(np.uint8)  # 0-1 arası -> 0-255 arası

    augmented = transform(image=image_uint8, mask=mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']

    # 🚀 Augmentation sonrası tekrar float32 yap
    aug_image = aug_image.astype(np.float32) / 255.0  # 0-255 -> 0-1 normalize tekrar

    return aug_image, aug_mask

