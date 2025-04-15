import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def visualize_mha_mask(mask_path, slice_index=None, cmap="nipy_spectral"):
    mask = sitk.ReadImage(mask_path)
    mask_np = sitk.GetArrayFromImage(mask).astype(np.uint8)  # [D, H, W]

    if slice_index is None:
        slice_index = mask_np.shape[0] // 2

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_np[slice_index], cmap="tab20", vmin=0, vmax=48)

    plt.title(f"Mask: {mask_path.split('/')[-1]} | Slice {slice_index}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"🎯 Mask shape: {mask_np.shape}")
    print(f"🧩 Sınıflar: {np.unique(mask_np)}")

# 🔁 Script olarak çalıştırıldığında
if __name__ == "__main__":
    # Sabit örnek — istersen argparse ile dışarıdan da alabiliriz
    visualize_mha_mask(
        mask_path="prediction_004.mha",
        slice_index=30
    )
