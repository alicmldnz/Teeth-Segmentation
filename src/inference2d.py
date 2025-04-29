import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model_2d import UNet2D
import torch.nn.functional as F

TARGET_CLASSES = [16, 17, 18, 26, 27, 28]  # Segment edeceğimiz sınıflar
NUM_CLASSES = 7  # 0 = background + 6 sınıf

def remap_mask(mask):
    """Mask değerlerini 16,17,18,26,27,28 → 1,2,3,4,5,6 olarak yeniden haritalar."""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for idx, cls in enumerate(TARGET_CLASSES, start=1):
        new_mask[mask == cls] = idx
    return new_mask

def inference_all_test(model_path, test_images_dir, test_masks_dir, save_dir="None", device="cuda"):
    if save_dir != "None":
        os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = UNet2D(in_channels=1, out_channels=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Sadece "upper" slice dosyaları seçiyoruz
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".npy") and "upper" in f])

    dice_scores = []
    accuracies = []

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(test_images_dir, image_file)
        mask_path = os.path.join(test_masks_dir, image_file)

        image = np.load(image_path)
        mask = np.load(mask_path)
        mask = remap_mask(mask)  # 🔥 yeniden etiketle

        # Görselleştirme için normalize
        image_disp = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]

        with torch.no_grad():
            output = model(img_tensor)  # [1, NUM_CLASSES, H, W]
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)  # [H, W]

        pred_np = pred.cpu().numpy()

        # Dice Score Hesapla
        dice = 0.0
        for cls in range(1, NUM_CLASSES):
            pred_cls = (pred_np == cls).astype(np.uint8)
            mask_cls = (mask == cls).astype(np.uint8)
            intersection = np.sum(pred_cls * mask_cls)
            union = np.sum(pred_cls) + np.sum(mask_cls)
            dice += (2 * intersection + 1e-6) / (union + 1e-6)
        dice /= (NUM_CLASSES - 1)  # Background hariç sınıflar
        dice_scores.append(dice)

        # Accuracy Hesapla
        correct = np.sum(pred_np == mask)
        total = mask.size
        acc = correct / total
        accuracies.append(acc)

        # İlk 5 örneği kaydet veya göster
        if idx < 5:
            plt.figure(figsize=(15,5))

            plt.subplot(1,3,1)
            plt.imshow(image_disp, cmap="gray")
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            plt.title("Ground Truth Mask")
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.imshow(pred_np, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            plt.title(f"Prediction - Dice: {dice:.4f}")
            plt.axis("off")

            plt.tight_layout()

            if save_dir != "None":
                plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"))
                plt.close()
            else:
                plt.show()

    # Ortalama Metrikler
    avg_dice = np.mean(dice_scores)
    avg_acc = np.mean(accuracies)

    print(f"📈 Ortalama Dice Score: {avg_dice:.4f}")
    print(f"📈 Ortalama Accuracy: {avg_acc:.4f}")

    return avg_dice, avg_acc

if __name__ == "__main__":
    inference_all_test(
        model_path="best_model.pth",
        test_images_dir=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\test\images",
        test_masks_dir=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\test\masks",
        save_dir=r"C:\Users\user\Desktop\TeethSegmentation\inference_results",
        device="cuda"
    )
