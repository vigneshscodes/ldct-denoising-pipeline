# ============================
# METRICS EVALUATION (FINAL PERFECT)
# ============================

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ============================
# PATHS
# ============================

LDCT_ROOT   = r"D:\CT_Datasets\LDCT"
SEG_ROOT    = r"D:\CT_Datasets\Segmentation"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_Output"

PATIENT = "LIDC-IDRI-0001"

# ============================
# UTIL
# ============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


# ============================
# METRIC FUNCTION (FINAL FIXED)
# ============================

def compute_metrics(gt, pred, mask):

    if np.sum(mask) == 0:
        return None, None

    # PSNR (lung only)
    gt_masked = gt[mask == 1]
    pred_masked = pred[mask == 1]
    psnr_val = psnr(gt_masked, pred_masked, data_range=1.0)

    # SSIM (masked properly)
    gt_mask = gt * mask
    pred_mask = pred * mask
    ssim_val = ssim(gt_mask, pred_mask, data_range=1.0)

    return psnr_val, ssim_val


# ============================
# MAIN LOOP
# ============================

psnr_ldct_list, ssim_ldct_list = [], []
psnr_region_list, ssim_region_list = [], []

for root, dirs, files in os.walk(LDCT_ROOT):

    if PATIENT not in root:
        continue

    relative_path = os.path.relpath(root, LDCT_ROOT)

    seg_root = os.path.join(SEG_ROOT, relative_path)
    den_root = os.path.join(PHASE2_ROOT, relative_path)

    for file in files:

        if not file.endswith("_ndct.png"):
            continue

        base_name = file.replace("_ndct.png", "")

        ndct_path = os.path.join(root, base_name + "_ndct.png")
        ldct_path = os.path.join(root, base_name + "_ldct.png")
        lung_path = os.path.join(seg_root, base_name + "_lung_mask.png")
        region_path = os.path.join(den_root, base_name + "_region.png")

        if not all(os.path.exists(p) for p in [ndct_path, ldct_path, lung_path, region_path]):
            continue

        print(f"Evaluating {base_name}")

        ndct = load_image(ndct_path)
        ldct = load_image(ldct_path)
        region = load_image(region_path)
        mask = load_image(lung_path)

        if ndct is None or ldct is None or region is None or mask is None:
            continue

        mask = (mask > 0.5).astype(np.uint8)

        # 🔥 NEW FIX: erode mask to avoid boundary inflation
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        # Skip small lung slices
        if np.sum(mask) / mask.size < 0.05:
            continue

        # LDCT
        psnr_ldct, ssim_ldct = compute_metrics(ndct, ldct, mask)

        # REGION
        psnr_region, ssim_region = compute_metrics(ndct, region, mask)

        if psnr_ldct is not None:
            psnr_ldct_list.append(psnr_ldct)
            ssim_ldct_list.append(ssim_ldct)

        if psnr_region is not None:
            psnr_region_list.append(psnr_region)
            ssim_region_list.append(ssim_region)


# ============================
# FINAL RESULTS
# ============================

def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0

print("\n==============================")
print("FINAL RESULTS (LUNG REGION)")
print("==============================")

print(f"LDCT      → PSNR: {safe_mean(psnr_ldct_list):.2f}, SSIM: {safe_mean(ssim_ldct_list):.4f}")
print(f"REGION    → PSNR: {safe_mean(psnr_region_list):.2f}, SSIM: {safe_mean(ssim_region_list):.4f}")

print("\nImprovement over LDCT:")
print(f"PSNR Gain : {safe_mean(psnr_region_list) - safe_mean(psnr_ldct_list):.2f}")
print(f"SSIM Gain : {safe_mean(ssim_region_list) - safe_mean(ssim_ldct_list):.4f}")