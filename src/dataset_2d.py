import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Hedef diş sınıfları ve ignore index
target_classes = [16, 17, 18, 26, 27, 28]
ignore_index = 255

class ToothSegmentation2DDataset(Dataset):
    def __init__(self, images_dir, masks_dir, only_upper=True, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        image_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".npy"))
        mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".npy"))

        if only_upper:
            image_files = [f for f in image_files if "upper" in f]
            mask_files = [f for f in mask_files if "upper" in f]

        assert len(image_files) == len(mask_files), "Image/mask count mismatch"
        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_np = np.load(os.path.join(self.images_dir, self.image_files[idx]))
        msk_np = np.load(os.path.join(self.masks_dir, self.mask_files[idx]))

        # Remap mask values: default ignore, background 0, target teeth 1-6
        new_mask = np.full_like(msk_np, ignore_index, dtype=np.uint8)
        new_mask[msk_np == 0] = 0
        for i, cls in enumerate(target_classes, start=1):
            new_mask[msk_np == cls] = i

        # Normalize image to [0,1]
        img_np = img_np.astype(np.float32)
        if img_np.max() > img_np.min():
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Apply transform if provided
        if self.transform:
            try:
                # Albumentations-style transform
                data = self.transform(image=img_np, mask=new_mask)
                img_tensor = data['image']
                mask_tensor = data['mask'].long()
            except TypeError:
                # Positional transform function
                img_tensor, mask_tensor = self.transform(img_np, new_mask)
        else:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            mask_tensor = torch.from_numpy(new_mask).long()

        return img_tensor, mask_tensor
