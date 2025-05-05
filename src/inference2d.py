import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Eğer UNet2D yerine SMP Unet kullandıysanız, model_2d.UNet2D değil SMP’yi içe alın

TARGET_CLASSES = [16, 17, 18, 26, 27, 28]
NUM_CLASSES = 7  # background + 6 hedef sınıf

def remap_mask(mask: np.ndarray) -> np.ndarray:
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for idx, cls in enumerate(TARGET_CLASSES, start=1):
        new_mask[mask == cls] = idx
    return new_mask

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    # Eğitimde kullanılan tam aynı mimari:
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,           # kendi kaydettiğiniz ağırlıkları yüklüyoruz
        in_channels=1,
        classes=NUM_CLASSES,
        decoder_attention_type="scse"
    )
    # Güvenlik uyarısını bastırmak için weights_only=True kullanabilirsiniz (Torch >=2.1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

def inference_all_test(
    model_path: str,
    test_images_dir: str,
    test_masks_dir: str,
    save_dir: str = None,
    device: str = "cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(test_images_dir)
        if f.endswith(".npy") and "upper" in f
    ])

    dice_scores = []
    accuracies = []

    for idx, image_file in enumerate(image_files):
        # Yükleme
        image = np.load(os.path.join(test_images_dir, image_file))
        mask_orig = np.load(os.path.join(test_masks_dir, image_file))
        mask = remap_mask(mask_orig)

        # Görüntü normalizasyonu
        disp = (image - image.min()) / (image.max() - image.min() + 1e-8)

        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(inp)
            pred = torch.argmax(logits.softmax(dim=1), dim=1).squeeze(0).cpu().numpy()

        # Dice hesapla
        dice = 0.0
        for cls in range(1, NUM_CLASSES):
            p = (pred == cls).astype(np.uint8)
            t = (mask == cls).astype(np.uint8)
            inter = np.sum(p * t)
            uni = np.sum(p) + np.sum(t)
            dice += (2 * inter + 1e-6) / (uni + 1e-6)
        dice /= (NUM_CLASSES - 1)
        dice_scores.append(dice)

        # Accuracy
        acc = np.sum(pred == mask) / mask.size
        accuracies.append(acc)

        # İlk 5’ini kaydet veya göster
        if idx < 5:
            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.imshow(disp, cmap="gray"); plt.title("Image"); plt.axis("off")
            plt.subplot(1,3,2)
            plt.imshow(mask, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            plt.title("Ground Truth"); plt.axis("off")
            plt.subplot(1,3,3)
            plt.imshow(pred, cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES-1)
            plt.title(f"Prediction\nDice: {dice:.4f}"); plt.axis("off")
            plt.tight_layout()

            if save_dir:
                out_name = os.path.splitext(image_file)[0] + ".png"
                plt.savefig(os.path.join(save_dir, out_name), bbox_inches="tight")
                plt.close()
            else:
                plt.show()

    avg_dice = float(np.mean(dice_scores))
    avg_acc = float(np.mean(accuracies))
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
