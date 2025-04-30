from train_2d import train_model

def main():
    # Eğitim parametreleri
    train_images_dir = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\images"
    train_masks_dir = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\train\masks"
    val_images_dir = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\val\images"
    val_masks_dir = r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\val\masks"

    batch_size = 8
    lr = 1e-4
    epochs = 50
    device = "cuda"  # veya "cpu" kullanabilirsin

    print("🚀 Eğitim başlatılıyor...")
    train_model(
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        val_images_dir=val_images_dir,
        val_masks_dir=val_masks_dir,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=device,
        pretrained_model_path=None
        
    )

    print("✅ Eğitim tamamlandı!")

if __name__ == "__main__":
    main()
