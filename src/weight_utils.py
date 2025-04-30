import os
import numpy as np
import torch

# Orijinal mask etiketleri
TARGET_CLASSES = [16, 17, 18, 26, 27, 28]
NUM_CLASSES = 7  # 0: background, 1–6: TARGET_CLASSES sırasıyla

def compute_class_weights_from_labels(
    labels_dir: str,
    num_classes: int = NUM_CLASSES,
    save_path: str = None,
    only_upper: bool = True
) -> torch.Tensor:
    """
    labels_dir içindeki tüm .npy mask dosyalarındaki
      - background piksellerini (TARGET_CLASSES dışında kalan) index 0'a,
      - 16,17,18,26,27,28 piksellerini ise sırasıyla 1–6 index'lerine sayar.
    only_upper=True ise sadece dosya adı içinde "upper" geçen maskeleri sayar.
    Ardından her sınıf için 1/(piksel sayısı) formülüne göre ağırlık hesaplar,
    normalize eder, ekrana bastırır ve isteğe bağlı kaydeder.
    """
    # 1) Dosya listesi
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".npy")]
    if only_upper:
        label_files = [f for f in label_files if "upper" in f]

    if not label_files:
        raise RuntimeError(f"No mask files found in {labels_dir!r} (only_upper={only_upper})")

    # 2) Piksel sayacı
    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    # 3) Her dosya için say
    for fname in label_files:
        mask = np.load(os.path.join(labels_dir, fname))

        # background: TARGET_CLASSES dışında kalan tüm pikseller
        bg_mask = ~np.isin(mask, TARGET_CLASSES)
        pixel_counts[0] += np.sum(bg_mask)

        # her hedef sınıf
        for idx, cls in enumerate(TARGET_CLASSES, start=1):
            pixel_counts[idx] += np.sum(mask == cls)

    # 4) Piksel sayılarını yazdır
    print("\n📊 Class Pixel Counts:")
    print(f"  Class 0 (background): {pixel_counts[0]} pixels")
    for idx, cls in enumerate(TARGET_CLASSES, start=1):
        print(f"  Class {cls} → index {idx}: {pixel_counts[idx]} pixels")

    # 5) Ağırlıkları ters frekansla hesapla ve normalize et
    class_weights = 1.0 / (pixel_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # 6) Ağırlıkları yazdır
    print("\n⚖️ Final Class Weights:")
    print(f"  Index 0 (background): {class_weights_tensor[0]:.6f}")
    for idx, cls in enumerate(TARGET_CLASSES, start=1):
        print(f"  Class {cls} → index {idx}: {class_weights_tensor[idx]:.6f}")

    # 7) İsteğe bağlı kaydet
    if save_path:
        torch.save(class_weights_tensor, save_path)
        print(f"\n💾 Class weights saved to {save_path!r}")

    return class_weights_tensor


if __name__ == "__main__":
    # Örnek kullanım
    labels_directory = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\masks"
    weights = compute_class_weights_from_labels(
        labels_dir=labels_directory,
        num_classes=NUM_CLASSES,
        save_path="class_weights.pt",
        only_upper=True   # sadece upper maskeleri say
    )
