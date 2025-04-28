import torch
import numpy as np
import matplotlib.pyplot as plt
from model_2d import UNet2D

def inference_single_image(model_path, image_path, device="cuda", num_classes=49):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Modeli yükle
    model = UNet2D(in_channels=1, out_channels=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Görüntüyü yükle
    image = np.load(image_path)  # [H, W]
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]

    with torch.no_grad():
        output = model(image)  # [1, num_classes, H, W]
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)  # [1, H, W]

    prediction = pred.squeeze(0).cpu().numpy()  # [H, W]

    return prediction

def show_prediction_with_ground_truth(image_path, prediction, mask_path):
    image = np.load(image_path)
    ground_truth = np.load(mask_path)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap="nipy_spectral")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="nipy_spectral")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"🎯 Ground Truth Sınıflar: {np.unique(ground_truth)}")
    print(f"🎯 Prediction Sınıflar: {np.unique(prediction)}")


if __name__ == "__main__":
    pred = inference_single_image(
    model_path="best_model.pth",
    image_path=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\test\images\ToothFairy2P_049_0000_upper_slice_0.npy",
    device="cuda",
    num_classes=49
    )
    show_prediction_with_ground_truth(
    image_path=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\test\images\ToothFairy2P_049_0000_upper_slice_0.npy",
    prediction=pred,
    mask_path=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\test\masks\ToothFairy2P_049_0000_upper_slice_0.npy"
    )