import json
import matplotlib.pyplot as plt

def plot_metrics(json_path="metrics.json"):
    # Veriyi oku
    with open(json_path, "r") as f:
        data = json.load(f)

    # Epoch sayısı kadar boş liste
    epochs = [entry["epoch"] for entry in data]
    train_loss = [entry["train_loss"] for entry in data]
    val_loss = [entry["val_loss"] for entry in data]
    train_dice = [entry["train_dice"] for entry in data]
    val_dice = [entry["val_dice"] for entry in data]

    # 🔷 Loss Grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, '-o', label="Train Loss")
    plt.plot(epochs, val_loss, '-s', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png")  # opsiyonel: kaydet
    plt.show()

    # 🔷 Dice Score Grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_dice, '-o', label="Train Dice")
    plt.plot(epochs, val_dice, '-s', label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Train vs Validation Dice Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dice_plot.png")  # opsiyonel: kaydet
    plt.show()
if __name__ == "__main__":
    plot_metrics("metrics.json")