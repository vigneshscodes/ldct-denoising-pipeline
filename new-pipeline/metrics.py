# metrics.py

import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


# ============================
# CONFIGURE THESE PATHS
# ============================

NDCT_ROOT = r"D:\CT_Datasets\LDCT"        # Folder containing *_ndct.png
ENH_ROOT  = r"D:\CT_Datasets\LDCT_Enhanced"  # Folder containing enhanced + masks

OUTPUT_CSV = "baseline_metrics.csv"


# ============================
# Metric Functions
# ============================

def compute_psnr(gt, pred, mask):
    gt_lung = gt[mask == 1]
    pred_lung = pred[mask == 1]

    if len(gt_lung) == 0:
        return None

    mse = np.mean((gt_lung - pred_lung) ** 2)

    if mse == 0:
        return 100.0

    return 10 * np.log10(1.0 / mse)


def compute_ssim(gt, pred, mask):
    gt_masked = gt * mask
    pred_masked = pred * mask

    return structural_similarity(
        gt_masked,
        pred_masked,
        data_range=1.0
    )


# ============================
# Utility Functions
# ============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return img


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    return mask


# ============================
# MAIN EVALUATION LOOP
# ============================

psnr_ldct_list = []
psnr_enh_list = []
ssim_ldct_list = []
ssim_enh_list = []

for root, dirs, files in os.walk(NDCT_ROOT):
    for file in files:

        if not file.endswith("_ndct.png"):
            continue

        base_name = file.replace("_ndct.png", "")

        ndct_path = os.path.join(root, file)
        ldct_path = os.path.join(root, base_name + "_ldct.png")

        # Enhanced + mask stored in ENH_ROOT
        relative_path = os.path.relpath(root, NDCT_ROOT)
        enh_folder = os.path.join(ENH_ROOT, relative_path)

        enh_path = os.path.join(enh_folder, base_name + "_enhanced.png")
        mask_path = os.path.join(enh_folder, base_name + "_lung_mask.png")

        if not os.path.exists(enh_path) or not os.path.exists(mask_path):
            continue

        # Load images
        ndct = load_image(ndct_path)
        ldct = load_image(ldct_path)
        enh  = load_image(enh_path)
        mask = load_mask(mask_path)

        # Skip slices with very small lung area
        lung_percent = np.sum(mask) / mask.size
        if lung_percent < 0.05:
            continue

        # Compute metrics
        psnr_ldct = compute_psnr(ndct, ldct, mask)
        psnr_enh  = compute_psnr(ndct, enh, mask)

        ssim_ldct = compute_ssim(ndct, ldct, mask)
        ssim_enh  = compute_ssim(ndct, enh, mask)

        if psnr_ldct is not None:
            psnr_ldct_list.append(psnr_ldct)
            psnr_enh_list.append(psnr_enh)
            ssim_ldct_list.append(ssim_ldct)
            ssim_enh_list.append(ssim_enh)

# ============================
# FINAL RESULTS
# ============================

print("\n===== Baseline Results (Lung Region) =====")
print(f"LDCT  PSNR: {np.mean(psnr_ldct_list):.3f} ± {np.std(psnr_ldct_list):.3f}")
print(f"ENH   PSNR: {np.mean(psnr_enh_list):.3f} ± {np.std(psnr_enh_list):.3f}")
print(f"LDCT  SSIM: {np.mean(ssim_ldct_list):.4f} ± {np.std(ssim_ldct_list):.4f}")
print(f"ENH   SSIM: {np.mean(ssim_enh_list):.4f} ± {np.std(ssim_enh_list):.4f}")

# Save to CSV
df = pd.DataFrame({
    "PSNR_LDCT": psnr_ldct_list,
    "PSNR_ENH": psnr_enh_list,
    "SSIM_LDCT": ssim_ldct_list,
    "SSIM_ENH": ssim_enh_list
})

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved detailed metrics to {OUTPUT_CSV}")