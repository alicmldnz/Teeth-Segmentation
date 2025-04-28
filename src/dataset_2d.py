import os
import torch
import numpy as np
from torch.utils.data import Dataset

class ToothSegmentation2DDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npy")])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.images_dir, self.image_files[idx]))
        mask = np.load(os.path.join(self.masks_dir, self.mask_files[idx]))

        if self.transform:
            image, mask = self.transform(image, mask)
    
        image = torch.from_numpy(image).unsqueeze(0).float()  # [1, H, W]
        mask = torch.from_numpy(mask).long()                 # [H, W]
    
        return image, mask

