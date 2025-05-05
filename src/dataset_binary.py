import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ToothBinaryDataset(Dataset):
    """
    Binary diş segmentasyonu:
     - mask!=0 ise diş (1), mask==0 ise background (0).
    Yüksek oranda background olduğu için ignore_index yok.
    """
    def __init__(self, images_dir, masks_dir, only_upper=True, transform=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform

        # .npy slice dosyaları
        imgs = sorted(f for f in os.listdir(images_dir) if f.endswith(".npy"))
        msks = sorted(f for f in os.listdir(masks_dir ) if f.endswith(".npy"))
        if only_upper:
            imgs = [f for f in imgs if "upper" in f]
            msks = [f for f in msks if "upper" in f]
        assert len(imgs)==len(msks), "counts mismatch"
        self.image_files = imgs
        self.mask_files  = msks

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.images_dir, self.image_files[idx]))
        msk = np.load(os.path.join(self.masks_dir,  self.mask_files[idx]))

        # binary mask
        bin_mask = (msk != 0).astype(np.uint8)  

        # normalize image 0–1
        if img.max()>img.min():
            img = (img - img.min())/(img.max()-img.min())

        # augment / transform
        if self.transform:
            img, bin_mask = self.transform(img, bin_mask)

        # to tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()    # [1,H,W]
        msk_tensor = torch.from_numpy(bin_mask).long()             # [H,W] values 0/1

        return img_tensor, msk_tensor
