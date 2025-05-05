import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp

from dataset_2d import ToothSegmentation2DDataset
from weight_utils import compute_class_weights_from_labels
from losses import CombinedLoss
from train_logger import TrainLogger
from early_stopping import EarlyStopping
from transforms import get_training_augmentation, apply_augmentation


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    accuracies = []

    with torch.no_grad():
        for img_batch, label_batch in loader:
            img_batch   = img_batch.to(device)            # [B,1,H,W]
            label_batch = label_batch.to(device)          # [B,H,W]

            logits = model(img_batch)                     # [B,num_classes,H,W]
            loss   = criterion(logits, label_batch)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)            # [B,H,W]

            # ignore_index maskesi
            valid = (label_batch != 255)

            # per-class Dice
            dice = 0.0
            for cls in range(1, num_classes):
                p = ((pred == cls) & valid).float()
                t = ((label_batch == cls) & valid).float()
                inter = (p * t).sum()
                union = p.sum() + t.sum()
                dice += (2.0 * inter + 1e-6) / (union + 1e-6)
            dice = dice / (num_classes - 1)
            dice_scores.append(dice.item())

            # overall accuracy on valid pixels
            correct = ((pred == label_batch) & valid).sum()
            total_valid = valid.sum()
            acc = correct.float() / (total_valid.float() + 1e-6)
            accuracies.append(acc.item())

    avg_loss = total_loss / len(loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_acc  = sum(accuracies) / len(accuracies)
    return avg_loss, avg_dice, avg_acc


def train_model(
    train_images_dir: str,
    train_masks_dir:  str,
    val_images_dir:   str,
    val_masks_dir:    str,
    num_classes:      int    = 7,
    batch_size:       int    = 8,
    lr:               float  = 1e-4,
    epochs:           int    = 50,
    device:           str    = "cuda",
    weight_save_path: str    = "class_weights.pt",
    pretrained_model_path: str = None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # --- Dataset & DataLoader ---
    train_dataset = ToothSegmentation2DDataset(
        train_images_dir,
        train_masks_dir,
        only_upper=True,
        transform=None
        # lambda img, msk: apply_augmentation(img, msk, get_training_augmentation())
    )
    val_dataset = ToothSegmentation2DDataset(
        val_images_dir,
        val_masks_dir,
        only_upper=True,
        transform=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # --- Model ---
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
        decoder_attention_type="scse"
    ).to(device)

    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"✅ Pretrained model loaded: {pretrained_model_path}")

    # --- Loss & Optimizer & Utilities ---
    class_weights = compute_class_weights_from_labels(
        labels_dir=train_masks_dir,
        num_classes=num_classes,
        save_path=weight_save_path,
        only_upper=True
    ).to(device)

    criterion = CombinedLoss(
        weight=class_weights,
        ce_weight=1.0,
        dice_weight=1.0,
        ignore_index=255
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler    = torch.amp.GradScaler('cuda')
    logger    = TrainLogger("metrics.json")
    early_stopper = EarlyStopping(patience=10)

    best_val_dice = 0.0

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        print(f"\n📚 Epoch {epoch}/{epochs}")
        model.train()
        running_loss = 0.0
        train_dices = []

        for img_batch, label_batch in train_loader:
            img_batch   = img_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(img_batch)
                loss   = criterion(logits, label_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Training Dice (per-batch)
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                valid = (label_batch != 255)
                dice = 0.0
                for cls in range(1, num_classes):
                    p = ((pred == cls) & valid).float()
                    t = ((label_batch == cls) & valid).float()
                    inter = (p * t).sum()
                    union = p.sum() + t.sum()
                    dice += (2.0 * inter + 1e-6) / (union + 1e-6)
                dice = dice / (num_classes - 1)
                train_dices.append(dice.item())

        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = sum(train_dices) / len(train_dices)
        print(f"🔥 Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f}")
        # --- Validation ---
        val_loss, val_dice, val_acc = evaluate(model, val_loader, criterion, device, num_classes)
        print(f"✅ Val Loss: {val_loss:.4f} | 🎯 Val Dice: {val_dice:.4f} | 📊 Val Acc: {val_acc:.4f}")
        logger.log(epoch, avg_train_loss, val_loss, avg_train_dice, val_dice)

        # --- Checkpoint & Early Stop ---
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Best model saved!")

        early_stopper(val_dice)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

    return model


if __name__ == "__main__":
    train_model(
        train_images_dir      = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\images",
        train_masks_dir       = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\masks",
        val_images_dir        = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\val\images",
        val_masks_dir         = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\val\masks",
        num_classes           = 7,
        batch_size            = 8,
        lr                    = 1e-4,
        epochs                = 50,
        device                = "cuda",
        weight_save_path      = "class_weights.pt",
        pretrained_model_path = None
    )
