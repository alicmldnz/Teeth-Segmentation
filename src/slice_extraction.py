import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def extract_best_slices_smart(images_dir, masks_dir, save_images_dir, save_masks_dir, target_shape=(410, 410)):
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_masks_dir, exist_ok=True)

    upper_teeth = {11, 12, 13, 14, 15, 16, 17, 18,
                   21, 22, 23, 24, 25, 26, 27, 28}
    lower_teeth = {31, 32, 33, 34, 35, 36, 37, 38,
                   41, 42, 43, 44, 45, 46, 47, 48}

    image_files = [f for f in os.listdir(images_dir) if f.endswith(".mha")]

    for file in tqdm(image_files, desc="Smart slice extraction"):
        image_path = os.path.join(images_dir, file)
        mask_path = os.path.join(masks_dir, file.replace("_0000", ""))

        image_3d = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # [Z, H, W]
        mask_3d = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))    # [Z, H, W]

        # Her slice için upper ve lower voxel sayıları
        upper_voxels = np.isin(mask_3d, list(upper_teeth)).sum(axis=(1, 2))
        lower_voxels = np.isin(mask_3d, list(lower_teeth)).sum(axis=(1, 2))

        has_upper = upper_voxels.sum() > 0
        has_lower = lower_voxels.sum() > 0

        base_filename = file.replace(".mha", "")

        if has_upper and has_lower:
            # İkisi de varsa → 2 slice
            best_upper_idx = np.argmax(upper_voxels)
            best_lower_idx = np.argmax(lower_voxels)

            save_slice(image_3d, mask_3d, best_upper_idx, save_images_dir, save_masks_dir, base_filename, "upper", target_shape)
            save_slice(image_3d, mask_3d, best_lower_idx, save_images_dir, save_masks_dir, base_filename, "lower", target_shape)

        elif has_upper:
            best_upper_idx = np.argmax(upper_voxels)
            save_slice(image_3d, mask_3d, best_upper_idx, save_images_dir, save_masks_dir, base_filename, "upper", target_shape)

        elif has_lower:
            best_lower_idx = np.argmax(lower_voxels)
            save_slice(image_3d, mask_3d, best_lower_idx, save_images_dir, save_masks_dir, base_filename, "lower", target_shape)

def save_slice(image_3d, mask_3d, slice_idx, save_images_dir, save_masks_dir, base_filename, position, target_shape):
    image_slice = image_3d[slice_idx]
    mask_slice = mask_3d[slice_idx]

    print(f"📐 Orijinal slice shape: {image_slice.shape}")

    # Normalize
    if image_slice.max() - image_slice.min() > 1e-6:
        image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 1e-8)
    else:
        print(f"⚠️ Warning: Slice sabit intensity, normalize edilmedi.")

    print(f"📐 Normalize sonrası shape: {image_slice.shape}")

    # Padding veya crop
    image_slice = resize_or_pad_numpy(image_slice, target_shape, is_mask=False)
    mask_slice = resize_or_pad_numpy(mask_slice, target_shape, is_mask=True)

    print(f"📐 Padding sonrası shape: {image_slice.shape}")

    # 🔥 Shape kontrolü: 410x410 değilse kaydetme
    if image_slice.shape != target_shape or mask_slice.shape != target_shape:
        print(f"❌ Hatalı shape, kaydetmiyoruz: {base_filename}_{position}_slice_{slice_idx}")
        return

    # Save npy
    image_save_path = os.path.join(save_images_dir, f"{base_filename}_{position}_slice_{slice_idx}.npy")
    mask_save_path = os.path.join(save_masks_dir, f"{base_filename}_{position}_slice_{slice_idx}.npy")

    np.save(image_save_path, image_slice)
    np.save(mask_save_path, mask_slice)
    print(f"✅ Kaydedildi: {image_save_path}")

def resize_or_pad_numpy(array, target_shape, is_mask=False):
    import cv2
    h, w = array.shape
    th, tw = target_shape

    # 🔥 Yükseklik (h) için
    if h > th:
        start_h = (h - th) // 2
        array = array[start_h:start_h+th, :]
    elif h < th:
        pad_h = th - h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        array = np.pad(array, ((pad_top, pad_bottom), (0, 0)),
                       mode='constant', constant_values=0)

    # 🔥 Genişlik (w) için
    if w > tw:
        start_w = (w - tw) // 2
        array = array[:, start_w:start_w+tw]
    elif w < tw:
        pad_w = tw - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        array = np.pad(array, ((0, 0), (pad_left, pad_right)),
                       mode='constant', constant_values=0)

    return array



if __name__ == "__main__":
    extract_best_slices_smart(
        images_dir=r"C:\Users\user\Desktop\TeethSegmentation\ToothFairy2_DataSet\imagesTr",
        masks_dir=r"C:\Users\user\Desktop\TeethSegmentation\ToothFairy2_DataSet\labelsTr",
        save_images_dir=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\images",
        save_masks_dir=r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset\masks",
        target_shape=(410, 410)
    )