import os
import SimpleITK as sitk
import numpy as np
import torch

def compute_class_weights_from_labels(
    labels_dir,
    num_classes=49,
    epsilon=1e-6,
    save_path="class_weights.pt"
):
    # Eğer önceden kaydedilmişse → yükle
    if os.path.exists(save_path):
        print(f"📂 Kayıtlı ağırlıklar yüklendi: {save_path}")
        return torch.load(save_path)

    # Yeni ağırlık hesapla
    voxel_counts = np.zeros(num_classes, dtype=np.int64)
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".mha")]

    print(f"🔍 {len(label_files)} maske taranıyor...")

    for file in label_files:
        label_path = os.path.join(labels_dir, file)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.uint8)

        unique, counts = np.unique(label, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                voxel_counts[cls] += count

    inverse_freq = 1.0 / (voxel_counts + epsilon)
    weights = inverse_freq / inverse_freq.sum()

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    torch.save(weights_tensor, save_path)
    print(f"💾 Ağırlıklar kaydedildi: {save_path}")

    return weights_tensor
