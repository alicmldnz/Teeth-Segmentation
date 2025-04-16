class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_dice):
        score = val_dice

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f"⏳ EarlyStopping bekleme: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("🛑 EarlyStopping aktif! Eğitim durduruluyor.")
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
