import os
import torch
import numpy as np
from torch.utils.data import Dataset

TARGET_CLASS = 16  # 🔥 Sadece bu dişi segment edeceğiz!

class ToothSegmentation2DDataset(Dataset):
    def __init__(self, images_dir, masks_dir, only_upper=True, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npy")])

        if only_upper:
            image_files = [f for f in image_files if "upper" in f]
            mask_files = [f for f in mask_files if "upper" in f]

        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.images_dir, self.image_files[idx]))
        mask = np.load(os.path.join(self.masks_dir, self.mask_files[idx]))

        # 🔥 Maski binary hale getiriyoruz: 1 = target diş, 0 = background
        mask = (mask == TARGET_CLASS).astype(np.uint8)

        if self.transform:
            image, mask = self.transform(image, mask)

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()

        return image, mask



