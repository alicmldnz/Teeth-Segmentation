import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Hedef dişler
TARGET_CLASSES = [16, 17, 18, 26, 27, 28]
IGNORE_INDEX = 255

class ToothSegmentation2DDataset(Dataset):
    def __init__(self, images_dir, masks_dir, only_upper=True, transform=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform

        image_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".npy"))
        mask_files  = sorted(f for f in os.listdir(masks_dir ) if f.endswith(".npy"))

        if only_upper:
            image_files = [f for f in image_files if "upper" in f]
            mask_files  = [f for f in mask_files  if "upper" in f]

        self.image_files = image_files
        self.mask_files  = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.images_dir, self.image_files[idx]))
        msk = np.load(os.path.join(self.masks_dir,  self.mask_files[idx]))

        # 1) Başta tüm pikseller ignore
        new_mask = np.full_like(msk, IGNORE_INDEX, dtype=np.uint8)

        # 2) Gerçek background (0) → class 0
        new_mask[msk == 0] = 0

        # 3) Hedef dişler → 1..6
        for i, cls in enumerate(TARGET_CLASSES, start=1):
            new_mask[msk == cls] = i

        # 4) Augmentation
        if self.transform:
            img, new_mask = self.transform(img, new_mask)

        # 5) Tensor’a çevir
        img_tensor  = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(new_mask).long()

        return img_tensor, mask_tensor
