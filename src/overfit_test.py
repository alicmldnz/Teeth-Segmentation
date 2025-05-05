import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from train_2d import train_model  # veya model+dataset tanımlarını içe al
from dataset_2d import ToothSegmentation2DDataset
import segmentation_models_pytorch as smp
def overfit_test(
    images_dir, masks_dir,
    num_samples: int = 4,
    batch_size: int   = 4,
    lr: float         = 1e-4,
    iters: int        = 50,
    device: str       = "cuda"
):
    # 1) Dataset ve Subset
    base_ds = ToothSegmentation2DDataset(
        images_dir, masks_dir,
        only_upper=True,
        transform=None  # önce augment kapalı
    )
    idxs = list(range(min(num_samples, len(base_ds))))
    ds   = Subset(base_ds, idxs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2) Model + Loss + Optimizasyon
    model = smp.Unet(
        encoder_name="efficientnet-b3",        # ImageNet‐pretrained
        encoder_weights="imagenet",
        in_channels=1,                         # single channel input
        classes=7,                             # 0 + 6 tooth classes
        decoder_attention_type="scse"
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3) Overfit iterasyonu
    print(f"🔎 Overfit test: {len(idxs)} samples, {iters} iters")
    model.train()
    for i in range(1, iters+1):
        running = 0.0
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg = running / len(loader)
        if i % 5 == 0 or i == 1:
            print(f" Iter {i:02d}/{iters} — Loss: {avg:.4f}")

    print("✅ Overfit test tamamlandı.")

if __name__=="__main__":
    overfit_test(
        images_dir = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\images",
        masks_dir  = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\masks",
        num_samples=8,
        batch_size = 8,
        lr         = 1e-5,
        iters      = 500,
        device     = "cuda"
    )
