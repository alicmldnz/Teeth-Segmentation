import torch
import SimpleITK as sitk
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from model import UNet3D

def load_image(image_path, size=(128, 256, 256)):
    image = sitk.ReadImage(image_path)
    image_np = sitk.GetArrayFromImage(image).astype("float32")
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    image_tensor = F.interpolate(image_tensor, size=size, mode="trilinear", align_corners=False)
    return image_tensor, image  # hem tensor, hem orijinal itk-image

@torch.no_grad()
def run_inference(model_path, image_path, device="cuda", size=(128, 256, 256), slice_index=None, save_path=None):
    NUM_CLASSES = 49
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_tensor, original_itk = load_image(image_path, size)
    image_tensor = image_tensor.to(device)

    # 🔍 Diagnostik bilgi yazdır
    print("📐 Girdi tensör şekli:", image_tensor.shape)
    print("🔎 Tensör min:", image_tensor.min().item())
    print("🔎 Tensör max:", image_tensor.max().item())

    output = model(image_tensor)  # [1, C, D, H, W]
    pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()  # [D, H, W]

    print("📊 Tahmin edilen sınıflar:", np.unique(pred))

    if save_path:
        print(f"💾 Tahmin edilen maske kaydediliyor: {save_path}")
        pred_itk = sitk.GetImageFromArray(pred.astype(np.uint8))
        sitk.WriteImage(pred_itk, save_path)

if __name__ == "__main__":
    run_inference(
        model_path="best_model.pth",
        image_path=r"C:\Users\user\Desktop\TeethSegmentation\ToothFairy2_DataSet\imagesTr\ToothFairy2P_004_0000.mha",
        device="cuda",
        size=(128, 256, 256),
        save_path="prediction_004.mha"
    )
