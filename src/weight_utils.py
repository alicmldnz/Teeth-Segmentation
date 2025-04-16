import os
import numpy as np
import torch
import SimpleITK as sitk

def compute_class_weights_from_labels(
    labels_dir,
    num_classes=49,
    ignored_classes=[0, 1, 7],
    epsilon=1e-6,
    save_path="class_weights.pt",
    power=1.5
):
    if os.path.exists(save_path):
        print(f"📂 Kayıtlı class weight dosyası bulundu: {save_path}")
        return torch.load(save_path)

    print(f"🔍 Sınıf ağırlıkları hesaplanıyor...")

    voxel_counts = np.zeros(num_classes, dtype=np.int64)
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".mha")]

    for file in label_files:
        label_path = os.path.join(labels_dir, file)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.uint8)
        unique, counts = np.unique(label, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                voxel_counts[cls] += count

    print(f"📊 Voxel count per class:\n{voxel_counts}")

    # 0 olanlar ve kullanıcı tarafından ignore edilen sınıflar
    ignored_mask = (voxel_counts == 0)
    for cls in ignored_classes:
        ignored_mask[cls] = True

    # Ağırlıkları hesapla (inverse power)
    inverse_power = 1.0 / (voxel_counts + epsilon) ** power
    inverse_power[ignored_mask] = 0.0

    # Normalize: sadece kullanılacak sınıflar için
    fg = inverse_power[~ignored_mask]
    min_fg, max_fg = fg.min(), fg.max()
    scaled_fg = (fg - min_fg) / (max_fg - min_fg)  # [0,1]

    final_weights = np.zeros_like(inverse_power)
    final_weights[~ignored_mask] = scaled_fg
    final_weights[ignored_mask] = 0.0

    weights_tensor = torch.tensor(final_weights, dtype=torch.float32)
    torch.save(weights_tensor, save_path)

    print(f"✅ Final class weights:\n{weights_tensor}")
    return weights_tensor
