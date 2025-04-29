import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model_2d import UNet2D
import torch.nn.functional as F

def inference_all_test(model_path, test_images_dir, test_masks_dir, save_dir="None", device="cuda"):
    if save_dir != "None":
        os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = UNet2D(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Sadece "upper" içeren dosyaları seçiyoruz
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".npy") and "upper" in f])

    dice_scores = []
    accuracies = []

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(test_images_dir, image_file)
        mask_path = os.path.join(test_masks_dir, image_file)

        image = np.load(image_path)
        mask = np.load(mask_path)

        # Mask'i binary hale getir
        mask = (mask > 0).astype(np.uint8)

        # Görselleştirme için normalize
        image_disp = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        mask_disp = mask

        img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]

        with torch.no_grad():
            output = model(img_tensor)  # [1, 2, H, W]
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)  # [H, W]

        pred_np = pred.cpu().numpy()

        # Dice Score Hesapla
        intersection = np.sum((pred_np == 1) & (mask == 1))
        union = np.sum(pred_np == 1) + np.sum(mask == 1)
        dice = (2 * intersection) / (union + 1e-6)
        dice_scores.append(dice)

        # Accuracy Hesapla
        correct = np.sum(pred_np == mask)
        total = mask.size
        acc = correct / total
        accuracies.append(acc)

        # İlk 5 sample için görselleri kaydet
        if idx < 5:
            plt.figure(figsize=(15,5))
            
            plt.subplot(1,3,1)
            plt.imshow(image_disp, cmap="gray")
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(mask_disp, cmap="nipy_spectral", vmin=0, vmax=1)
            plt.title("Ground Truth Mask")
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.imshow(pred_np, cmap="nipy_spectral", vmin=0, vmax=1)
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
