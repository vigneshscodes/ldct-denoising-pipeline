# ============================
# METRICS EVALUATION (FINAL 100/100)
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
# METRIC COMPUTATION (LUNG ONLY)
# ============================

def compute_metrics(gt, pred, mask):

    if np.sum(mask) == 0:
        return None, None

    # Apply mask
    gt_masked = gt[mask == 1]
    pred_masked = pred[mask == 1]

    # PSNR (correct)
    psnr_val = psnr(gt_masked, pred_masked, data_range=1.0)

    # SSIM (better way — avoid zero padding effect)
    ssim_val = ssim(gt, pred, data_range=1.0)

    return psnr_val, ssim_val


# ============================
# MAIN LOOP
# ============================

psnr_ldct_list = []
ssim_ldct_list = []

psnr_bm3d_list = []
ssim_bm3d_list = []

psnr_den_list = []
ssim_den_list = []

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

        den_path  = os.path.join(den_root, base_name + "_region.png")
        bm3d_path = os.path.join(den_root, base_name + "_bm3d.png")

        if not all(os.path.exists(p) for p in [ndct_path, ldct_path, lung_path, den_path]):
            continue

        print(f"Evaluating {base_name}")

        ndct = load_image(ndct_path)
        ldct = load_image(ldct_path)
        den  = load_image(den_path)
        mask = load_image(lung_path)

        bm3d_img = load_image(bm3d_path) if os.path.exists(bm3d_path) else None

        if ndct is None or ldct is None or den is None or mask is None:
            continue

        mask = (mask > 0.5).astype(np.uint8)

        # Skip small lung slices
        if np.sum(mask) / mask.size < 0.05:
            continue

        # --------------------------
        # LDCT vs NDCT
        # --------------------------
        psnr_ldct, ssim_ldct = compute_metrics(ndct, ldct, mask)

        # --------------------------
        # BM3D vs NDCT (baseline)
        # --------------------------
        if bm3d_img is not None:
            psnr_bm3d, ssim_bm3d = compute_metrics(ndct, bm3d_img, mask)
        else:
            psnr_bm3d, ssim_bm3d = None, None

        # --------------------------
        # PROPOSED vs NDCT
        # --------------------------
        psnr_den, ssim_den = compute_metrics(ndct, den, mask)

        # Store results
        if psnr_ldct is not None:
            psnr_ldct_list.append(psnr_ldct)
            ssim_ldct_list.append(ssim_ldct)

        if psnr_bm3d is not None:
            psnr_bm3d_list.append(psnr_bm3d)
            ssim_bm3d_list.append(ssim_bm3d)

        if psnr_den is not None:
            psnr_den_list.append(psnr_den)
            ssim_den_list.append(ssim_den)


# ============================
# FINAL RESULTS
# ============================

def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0

print("\n==============================")
print("FINAL RESULTS (LUNG REGION)")
print("==============================")

print(f"LDCT      → PSNR: {safe_mean(psnr_ldct_list):.2f}, SSIM: {safe_mean(ssim_ldct_list):.4f}")
print(f"BM3D      → PSNR: {safe_mean(psnr_bm3d_list):.2f}, SSIM: {safe_mean(ssim_bm3d_list):.4f}")
print(f"PROPOSED  → PSNR: {safe_mean(psnr_den_list):.2f}, SSIM: {safe_mean(ssim_den_list):.4f}")

print("\nImprovement over LDCT:")
print(f"PSNR Gain : {safe_mean(psnr_den_list) - safe_mean(psnr_ldct_list):.2f}")
print(f"SSIM Gain : {safe_mean(ssim_den_list) - safe_mean(ssim_ldct_list):.4f}")