import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_roi_on_slice(image_npy_path: str, bbox: tuple):
    """
    image_npy_path: .npy olarak kaydedilmiş orijinal slice (2D) dosya yolu
    bbox: (x, y, w, h) formatındaki bounding‐box
    """
    # 1) Görüntüyü yükle ve normalize et
    img = np.load(image_npy_path)
    img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # 2) Bounding‐box koordinatlarını aç
    x, y, w, h = bbox

    # 3) Görüntüyü plot et ve ROI’yu çiz
    plt.figure(figsize=(6,6))
    plt.imshow(img_disp, cmap="gray")
    rect = plt.Rectangle((x, y), w, h, 
                         edgecolor="red", facecolor="none", linewidth=2)
    plt.gca().add_patch(rect)
    plt.title(f"ROI: x={x}, y={y}, w={w}, h={h}")
    plt.axis("off")
    plt.show()

# Kullanım örneği:
show_roi_on_slice(
    r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset/test/images/ToothFairy2F_009_0000_upper_slice_58.npy",
    (147, 18, 124, 45)
)
