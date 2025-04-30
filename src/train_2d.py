import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from model_2d import UNet2D
from dataset_2d import ToothSegmentation2DDataset
from early_stopping import EarlyStopping
from weight_utils import compute_class_weights_from_labels
from train_logger import TrainLogger
from transforms import get_training_augmentation, apply_augmentation
import os
from losses import CombinedLoss 
from torch.nn import CrossEntropyLoss

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train_model(
    train_images_dir,
    train_masks_dir,
    val_images_dir,
    val_masks_dir,
    num_classes=7,
    batch_size=8,
    lr=1e-4,
    epochs=50,
    device="cuda",
    weight_save_path="class_weights.pt",
    pretrained_model_path=None   # 🔥 yeni eklendi
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Dataset ve Dataloader
    train_dataset = ToothSegmentation2DDataset(
        train_images_dir, 
        train_masks_dir,
        transform=lambda img, msk: apply_augmentation(img, msk, get_training_augmentation())
        )
    val_dataset = ToothSegmentation2DDataset(val_images_dir, val_masks_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = UNet2D(in_channels=1, out_channels=num_classes).to(device)

    # Eğer pretrained ağırlık varsa yükle
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"✅ Pretrained model {pretrained_model_path} yüklendi!")

    # Loss
    weights = compute_class_weights_from_labels(
        labels_dir=train_masks_dir,
        num_classes=num_classes,
        save_path=weight_save_path,
        only_upper=True
    ).to(device)

    criterion = CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    logger = TrainLogger("metrics.json")
    early_stopper = EarlyStopping(patience=10)

    best_val_dice = 0.0

    # Eğitim döngüsü
    for epoch in range(epochs):
        print(f"\n📚 Epoch {epoch+1}/{epochs}")

        model.train()
        running_loss = 0.0
        train_dices = []

        for img_batch, label_batch in train_loader:

            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            
            """# Ignore sınıflar
            for cls in [0, 1, 7]:
                label_batch[label_batch == cls] = 255"""

            optimizer.zero_grad()

            with autocast('cuda'):
                output = model(img_batch)
                loss = criterion(output, label_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            with torch.no_grad():
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
                dice = 0.0
                for cls in range(1, num_classes):
                    pred_cls = (pred == cls).float()
                    target_cls = (label_batch == cls).float()
                    intersection = (pred_cls * target_cls).sum()
                    union = pred_cls.sum() + target_cls.sum()
                    dice += (2. * intersection + 1e-6) / (union + 1e-6)

                dice /= (num_classes - 1)
                train_dices.append(dice.item())

        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = sum(train_dices) / len(train_dices)

        val_loss, val_dice, val_acc = evaluate(model, val_loader, criterion, device, num_classes)

        print(f"✅ Val Loss: {val_loss:.4f} | 🎯 Val Dice: {val_dice:.4f} | 🎯 Val Acc: {val_acc:.4f}")

        logger.log(epoch+1, avg_train_loss, val_loss, avg_train_dice, val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Best model saved!")

        early_stopper(val_dice)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    accuracies = []

    with torch.no_grad():
        for img_batch, label_batch in loader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            output = model(img_batch)
            loss = criterion(output, label_batch)
            total_loss += loss.item()

            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

            # Dice
            dice = 0.0
            for cls in range(1, num_classes):
                pred_cls = (pred == cls).float()
                target_cls = (label_batch == cls).float()
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()
                dice += (2. * intersection + 1e-6) / (union + 1e-6)

            dice /= (num_classes - 1)
            dice_scores.append(dice.item())

            # 🔥 Accuracy
            correct = (pred == label_batch).float()
            acc = correct.sum() / correct.numel()
            accuracies.append(acc.item())

    avg_loss = total_loss / len(loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_acc = sum(accuracies) / len(accuracies)
    return avg_loss, avg_dice, avg_acc

