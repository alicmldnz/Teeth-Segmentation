import os
import shutil
import random

def split_2d_dataset(base_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

        for img in split_images:
            mask = img  # isim mask ile birebir aynı

            shutil.copy(os.path.join(image_dir, img), os.path.join(base_dir, split, "images", img))
            shutil.copy(os.path.join(mask_dir, mask), os.path.join(base_dir, split, "masks", mask))

    print("✅ Dataset split completed!")

if __name__ == "__main__":
    split_2d_dataset(
        base_dir=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
)