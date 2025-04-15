import argparse
from train import train_model
import torch, gc
def parse_args():
    parser = argparse.ArgumentParser(description="Diş Segmentasyon Eğitimi")

    parser.add_argument("--epochs", type=int, default=20, help="Eğitim epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    return parser.parse_args()

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()

    print("🚀 Diş Segmentasyon Eğitimi Başlatılıyor...")
    print(f"📌 Epochs: {args.epochs} | Batch size: {args.batch_size}")

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("✅ Eğitim tamamlandı.")