import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity


# ==========================
# PATHS
# ==========================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
ENH_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"
REGION_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"
CNN_ROOT = r"D:\CT_Datasets\Phase3_CNN_Refined"


# ==========================
# UTILITIES
# ==========================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


def compute_psnr(gt, pred, mask):

    gt = gt[mask == 1]
    pred = pred[mask == 1]

    mse = np.mean((gt - pred) ** 2)

    if mse == 0:
        return 100

    return 10 * np.log10(1.0 / mse)


def compute_ssim(gt, pred, mask):

    return structural_similarity(
        gt * mask,
        pred * mask,
        data_range=1.0
    )


# ==========================
# STORAGE
# ==========================

psnr_list = []
ssim_list = []


# ==========================
# LOOP
# ==========================

for root, dirs, files in os.walk(CNN_ROOT):

    relative = os.path.relpath(root, CNN_ROOT)

    for file in files:

        if not file.endswith("_cnn_refined.png"):
            continue

        base = file.replace("_cnn_refined.png", "")


        cnn_path = os.path.join(root, file)

        ndct_path = os.path.join(
            LDCT_ROOT,
            relative,
            base + "_ndct.png"
        )

        mask_path = os.path.join(
            ENH_ROOT,
            relative,
            base + "_lung_mask.png"
        )

        if not all(os.path.exists(p) for p in [ndct_path, mask_path]):
            continue


        cnn = load_image(cnn_path)
        ndct = load_image(ndct_path)
        mask = (load_image(mask_path) > 0.5).astype(np.uint8)


        psnr = compute_psnr(ndct, cnn, mask)
        ssim = compute_ssim(ndct, cnn, mask)

        psnr_list.append(psnr)
        ssim_list.append(ssim)


# ==========================
# FINAL RESULTS
# ==========================

print("\nCNN Refined Results\n")

print("PSNR:", np.mean(psnr_list), "±", np.std(psnr_list))
print("SSIM:", np.mean(ssim_list), "±", np.std(ssim_list))