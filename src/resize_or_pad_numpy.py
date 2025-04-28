import numpy as np
import cv2

def resize_or_pad_numpy(array, target_shape, is_mask=False):
    h, w = array.shape
    th, tw = target_shape

    # Eğer çok büyükse önce kırp (ama bu bizde olmaz gibi)
    if h > th or w > tw:
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        array = array[start_h:start_h+th, start_w:start_w+tw]
        return array

    # Padding miktarını hesapla
    pad_h = th - h
    pad_w = tw - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if is_mask:
        pad_value = 0  # mask için 0 (background)
    else:
        pad_value = 0  # görüntü için de siyah padding

    array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)),
                   mode='constant', constant_values=pad_value)
    
    return array
