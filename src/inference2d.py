import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model_2d import UNet2D

# Orijinal mask etiketleri → remap için
TARGET_CLASSES = [16, 17, 18, 26, 27, 28]
NUM_CLASSES = 7  # 0 = background + 6 hedef sınıf

def remap_mask(mask: np.ndarray) -> np.ndarray:
    """16,17,18,26,27,28 → 1–6 olarak mapler; geri kalanı 0."""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for idx, cls in enumerate(TARGET_CLASSES, start=1):
        new_mask[mask == cls] = idx
    return new_mask

def inference_all_test(
    model_path: str,
    test_images_dir: str,
    test_masks_dir: str,
    save_dir: str = "None",
    device: str = "cuda"
):
    if save_dir != "None":
        os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Model yükle
    model = UNet2D(in_channels=1, out_channels=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Sadece "upper" slice dosyaları
    image_files = sorted([
        f for f in os.listdir(test_images_dir)
        if f.endswith(".npy") and "upper" in f
    ])

    dice_scores = []
    accuracies = []

    for idx, image_file in enumerate(image_files):
        # Yol oluştur
        image_path = os.path.join(test_images_dir, image_file)
        mask_path  = os.path.join(test_masks_dir,  image_file)

        # Yükle
        image_orig = np.load(image_path)
        mask_orig  = np.load(mask_path)

        # Remap for metrics + display
        mask = remap_mask(mask_orig)

        # Normalize for display
        image_disp = (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min() + 1e-8)

        # Model input
        img_tensor = torch.from_numpy(image_orig).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        pred_np = pred.cpu().numpy()

        # --- Tahmin edilen sınıfları orijinal etiketlere dönüştür ve yazdır ---
        unique_idxs = np.unique(pred_np)
        pred_orig_classes = [TARGET_CLASSES[i-1] for i in unique_idxs if i > 0]
        print(f"Sample {idx} predicted classes: {pred_orig_classes}")

        # Dice Score (ortalama)
        dice = 0.0
        for cls_idx in range(1, NUM_CLASSES):
            p = (pred_np == cls_idx).astype(np.uint8)
            m = (mask   == cls_idx).astype(np.uint8)
            inter = np.sum(p * m)
            uni   = np.sum(p) + np.sum(m)
            dice += (2*inter + 1e-6) / (uni + 1e-6)
        dice /= (NUM_CLASSES - 1)
        dice_scores.append(dice)

        # Accuracy
        acc = np.sum(pred_np == mask) / mask.size
        accuracies.append(acc)

        # İlk 5 sample için görselleştir
        if idx < 5:
            plt.figure(figsize=(15,5))

            plt.subplot(1,3,1)
            plt.imshow(image_disp, cmap="gray")
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            plt.title("Ground Truth (remap)")
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.imshow(pred_np, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            plt.title(f"Prediction (remap)\nDice: {dice:.4f}")
            plt.axis("off")

            plt.tight_layout()
            if save_dir != "None":
                plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"))
                plt.close()
            else:
                plt.show()

    # Ortalama metrikler
    avg_dice = np.mean(dice_scores)
    avg_acc  = np.mean(accuracies)
    print(f"\n📈 Ortalama Dice: {avg_dice:.4f} | Ortalama Accuracy: {avg_acc:.4f}")

    return avg_dice, avg_acc


if __name__ == "__main__":
    inference_all_test(
        model_path      = "best_model.pth",
        test_images_dir = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\test\images",
        test_masks_dir  = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\test\masks",
        save_dir        = r"C:\Users\user\Desktop\TeethSegmentation\inference_results",
        device          = "cuda"
    )
