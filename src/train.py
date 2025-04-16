import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import ToothSegDataset
from model import UNet3D
from losses import get_loss
from metrics import dice_score
from visualize_prediction import visualize_sample
from tqdm import tqdm
from train_logger import TrainLogger
from weight_utils import compute_class_weights_from_labels
from torch.utils.data import Subset
from torch.amp import autocast, GradScaler
from early_stopping import EarlyStopping
from train_logger import TrainLogger


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes, scaler):
    model.train()
    total_loss = 0.0
    dice_scores = []

    for img_batch, label_batch in loader:
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device).squeeze(1).long()

        IGNORED_CLASSES = [0, 1, 7]
        for cls in IGNORED_CLASSES:
            label_batch[label_batch == cls] = 255

        optimizer.zero_grad()

        with autocast("cuda"):  # 🚀 mixed precision hesaplama
            output = model(img_batch)
            loss = criterion(output, label_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 🔍 Metrik: Dice hesapla (float32'e çevirmeye gerek yok çünkü argmax kullanıyoruz)
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
            dice_scores.append(dice.item())

    avg_loss = total_loss / len(loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    return avg_loss, avg_dice

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for img_batch, label_batch in loader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device).squeeze(1).long()

            IGNORED_CLASSES = [0, 1, 7]
            for cls in IGNORED_CLASSES:
                label_batch[label_batch == cls] = 255
            output = model(img_batch)
            loss = criterion(output, label_batch)

            total_loss += loss.item()
            dice = dice_score(output, label_batch, num_classes)
            dice_scores.append(dice.item())

    avg_loss = total_loss / len(loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    return avg_loss, avg_dice

def train_model(epochs, batch_size):
    import os
    # 🔧 Bellek ayarları
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    gc.collect()
    torch.cuda.empty_cache()

    NUM_CLASSES = 49
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    SIZE = (128, 256, 256)  # 🔁 Daha az bellek kullansın diye boyutu küçülttük Eski SIZE = (160, 352, 384)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger = TrainLogger("metrics.json")

    dataset = ToothSegDataset(
        images_dir=r"C:\Users\user\Desktop\TeethSegmentation\ToothFairy2_DataSet\imagesTr",
        labels_dir=r"C:\Users\user\Desktop\TeethSegmentation\ToothFairy2_DataSet\labelsTr",
        size=SIZE
    )
    total_size = len(dataset)
    # Yüzdelik oranlar
    train_ratio = 0.2
    val_ratio = 0.05
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))

    
    print(f"📊 Eğitim verisi: {len(train_dataset)} örnek")
    print(f"📊 Doğrulama verisi: {len(val_dataset)} örnek")
    """
     train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    """

    # 🧹 Daha az GPU baskısı için:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    torch.cuda.empty_cache()  # model.to(device) sonrası temizlik

    weights = compute_class_weights_from_labels(
    labels_dir=r"C:\Users\user\Desktop\TeethSegmentation\ToothFairy2_DataSet\labelsTr",
    num_classes=49,
    ignored_classes=[0, 1, 7],
    save_path="class_weights.pt"
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    # Optimizer & AMP
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda")

    # Logger & EarlyStopping
    logger = TrainLogger("metrics.json")
    early_stopper = EarlyStopping(patience=10)

    best_val_dice = 0.0

    for epoch in range(EPOCHS):
        print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, NUM_CLASSES, scaler)
        val_loss, val_dice = evaluate(model, val_loader, criterion, DEVICE, NUM_CLASSES)

        print(f"📉 Train Loss: {train_loss:.4f} | 🎯 Dice: {train_dice:.4f}")
        print(f"✅ Val   Loss: {val_loss:.4f} | 🎯 Dice: {val_dice:.4f}")

        logger.log(epoch+1, train_loss, val_loss, train_dice, val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 En iyi model kaydedildi!")
        
        early_stopper(val_dice)
        if early_stopper.early_stop:
            print("⏹️ Eğitim erken durduruldu.")
            break

        # 🧹 Epoch sonunda GPU belleğini temizle
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_model()
