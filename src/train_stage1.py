import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_2d import UNet2D
from dataset_binary import ToothBinaryDataset
from torch.cuda.amp import GradScaler, autocast

def train_stage1(
    train_images_dir, train_masks_dir,
    val_images_dir,   val_masks_dir,
    batch_size=8, lr=1e-4, epochs=50,
    device="cuda", save_model_path="stage1_teeth.pth"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Datasets & Loaders
    train_ds = ToothBinaryDataset(train_images_dir, train_masks_dir, only_upper=True, transform=None)
    val_ds   = ToothBinaryDataset(val_images_dir,   val_masks_dir,   only_upper=True, transform=None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, Loss, Opt
    model = UNet2D(in_channels=1, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")
    for ep in range(1, epochs+1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)         # [B,2,H,W]
                loss   = criterion(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        train_loss = running_loss/len(train_loader)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, msks).item()
        val_loss /= len(val_loader)

        print(f"Epoch {ep}/{epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model_path)
            print(" 💾 Saved best stage1 model")

    return model

if __name__=="__main__":
    # örnek çağrı
    train_stage1(
        r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\images",
        r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\masks",
        r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\val\images",
        r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\val\masks",
      batch_size=8, lr=1e-4, epochs=50,
      device="cuda", save_model_path="stage1_teeth.pth"
    )
