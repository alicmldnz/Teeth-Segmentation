import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch.nn.functional as F

class ToothSegDataset(Dataset):
    def __init__(self, images_dir, labels_dir, size=(160, 352, 384), transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.size = size
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.mha')])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.mha')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Dosyaları yükle
        image = sitk.ReadImage(os.path.join(self.images_dir, self.image_files[idx]))
        label = sitk.ReadImage(os.path.join(self.labels_dir, self.label_files[idx]))

        image = sitk.GetArrayFromImage(image).astype('float32')  # [D, H, W] olmalı
        label = sitk.GetArrayFromImage(label).astype('int64')    # [D, H, W] olmalı

        """"
        "# 🛡️ Güvenlik: Boyut kontrolü
        if image.ndim != 3:
            raise ValueError(f"❗ Hatalı görüntü: 3D değil, shape={image.shape}")

        if label.shape != image.shape:
            raise ValueError(f"❗ Maske ile görüntü aynı boyutta değil! image: {image.shape}, label: {label.shape}")

        if image.shape[0] < 3:
            raise ValueError(f"❗ Z-boyutu çok küçük: {image.shape[0]} slice")

        assert label.ndim == 3, f"Label 3D değil: {label.shape}"
        assert label.shape == image.shape, f"Label ve image boyutları eşleşmiyor: image={image.shape}, label={label.shape}"

        # Normalize et
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        """
        # Torch tensörüne çevir
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        # Resize (sabit boyuta getir)
        image = F.interpolate(image, size=self.size, mode='trilinear', align_corners=False).squeeze(0)
        label = F.interpolate(label.float(), size=self.size, mode='nearest').squeeze(0).long()

        return image, label

