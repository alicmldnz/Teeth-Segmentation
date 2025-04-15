import json
import os

class TrainLogger:
    def __init__(self, save_path="metrics_ft.json"):
        self.save_path = save_path
        self.history = []

        # Eğer eski bir log varsa, yükle
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                self.history = json.load(f)

    def log(self, epoch, train_loss, val_loss, train_dice, val_dice):
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice
        }
        self.history.append(entry)

        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def latest(self):
        return self.history[-1] if self.history else None
