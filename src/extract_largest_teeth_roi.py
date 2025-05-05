import numpy as np
import torch
import cv2
from model_2d import UNet2D

def extract_largest_teeth_roi(img_np, model, device="cuda"):
    """
    img_np: [H,W] normalized numpy slice
    model: stage1 UNet (in_channels=1,out_channels=2)
    returns: bbox (x,y,w,h) of largest connected component of teeth
    """
    # Forward
    model.to(device).eval()
    inp = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(inp)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]  # [H,W]

    # Binary mask
    teeth = (pred==1).astype(np.uint8)

    # CC analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(teeth, connectivity=8)
    # skip label 0 (background)
    if num_labels<=1:
        return None
    # find largest
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = np.argmax(areas) + 1
    x = stats[best, cv2.CC_STAT_LEFT]
    y = stats[best, cv2.CC_STAT_TOP]
    w = stats[best, cv2.CC_STAT_WIDTH]
    h = stats[best, cv2.CC_STAT_HEIGHT]
    return (x,y,w,h)

# örnek kullanım
if __name__=="__main__":
    import os
    # yükle model
    stage1 = UNet2D(1,2)
    stage1.load_state_dict(torch.load("stage1_teeth.pth"))
    # test slice
    img = np.load(r"C:\Users\user\Desktop\TeethSegmentation\2D_dataset/test/images/ToothFairy2F_009_0000_upper_slice_58.npy")
    # normalize
    img = (img-img.min())/(img.max()-img.min())
    bbox = extract_largest_teeth_roi(img, stage1)
    print("ROI bbox:", bbox)
