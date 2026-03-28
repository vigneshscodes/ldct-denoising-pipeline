# ============================
# METRICS FOR ENHANCEMENT STEP
# ============================

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ============================
# PATHS
# ============================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_Output"
ENH_ROOT = r"D:\CT_Datasets\Final_Enhanced"
SEG_ROOT = r"D:\CT_Datasets\Segmentation"

EVAL_PATIENT = "LIDC-IDRI-0001"

# ============================
# Utility
# ============================

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


# ============================
# MAIN
# ============================

psnr_ldct = []
ssim_ldct = []

psnr_region = []
ssim_region = []

psnr_enh = []
ssim_enh = []

for root, dirs, files in os.walk(LDCT_ROOT):

    if EVAL_PATIENT not in root:
        continue

    relative_path = os.path.relpath(root, LDCT_ROOT)
    seg_root = os.path.join(SEG_ROOT, relative_path)
    region_root = os.path.join(PHASE2_ROOT, relative_path)
    enh_root = os.path.join(ENH_ROOT, relative_path)

    for file in files:

        if not file.endswith("_ldct.png"):
            continue

        base = file.replace("_ldct.png", "")

        ndct_path = os.path.join(root, base + "_ndct.png")
        ldct_path = os.path.join(root, base + "_ldct.png")
        region_path = os.path.join(region_root, base + "_region.png")
        enh_path = os.path.join(enh_root, base + "_enhanced.png")
        lung_path = os.path.join(seg_root, base + "_lung_mask.png")

        if not all(os.path.exists(p) for p in [ndct_path, ldct_path, region_path, enh_path, lung_path]):
            continue

        ndct = load_img(ndct_path)
        ldct = load_img(ldct_path)
        region = load_img(region_path)
        enh = load_img(enh_path)
        lung = load_img(lung_path) > 0.5

        # apply lung mask
        ndct_l = ndct[lung]
        ldct_l = ldct[lung]
        region_l = region[lung]
        enh_l = enh[lung]

        # compute metrics
        psnr_ldct.append(psnr(ndct_l, ldct_l, data_range=1.0))
        ssim_ldct.append(ssim(ndct_l, ldct_l, data_range=1.0))

        psnr_region.append(psnr(ndct_l, region_l, data_range=1.0))
        ssim_region.append(ssim(ndct_l, region_l, data_range=1.0))

        psnr_enh.append(psnr(ndct_l, enh_l, data_range=1.0))
        ssim_enh.append(ssim(ndct_l, enh_l, data_range=1.0))


# ============================
# RESULTS
# ============================

print("\n========== FINAL METRICS ==========\n")

print("LDCT:")
print("PSNR:", np.mean(psnr_ldct))
print("SSIM:", np.mean(ssim_ldct))

print("\nRegion Denoised:")
print("PSNR:", np.mean(psnr_region))
print("SSIM:", np.mean(ssim_region))

print("\nEnhanced:")
print("PSNR:", np.mean(psnr_enh))
print("SSIM:", np.mean(ssim_enh))