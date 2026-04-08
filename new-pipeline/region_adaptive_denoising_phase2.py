# ============================
# REGION-WISE DENOISING (FINAL PERFECT)
# ============================

import os
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means
from bm3d import bm3d

# ============================
# CONFIG
# ============================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
SEG_ROOT  = r"D:\CT_Datasets\Segmentation"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_Output"

# --------------------------
# PATIENT SPLITS
# --------------------------
TRAIN_PATIENTS = [f"LIDC-IDRI-{i:04d}" for i in range(1, 21)]
VAL_PATIENTS   = [f"LIDC-IDRI-{i:04d}" for i in range(21, 24)]
TEST_PATIENTS  = [f"LIDC-IDRI-{i:04d}" for i in range(24, 27)]

ALL_PATIENTS = TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS

os.makedirs(PHASE2_ROOT, exist_ok=True)

# ============================
# UTIL
# ============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.clip(img, 0, 1)
    cv2.imwrite(path, (img * 255).astype(np.uint8))

# ============================
# FILTERS (FINAL TUNED)
# ============================

def apply_bilateral(img):
    return cv2.bilateralFilter(
        (img * 255).astype(np.uint8),
        d=7,
        sigmaColor=45,
        sigmaSpace=45
    ).astype(np.float32) / 255.0


def apply_nlm(img):
    return denoise_nl_means(
        img,
        h=0.15,
        fast_mode=True,
        patch_size=5,
        patch_distance=6
    ).astype(np.float32)


def apply_bm3d(img):
    return bm3d(img, sigma_psd=0.05).astype(np.float32)

# ============================
# MAIN LOOP
# ============================

for root, dirs, files in os.walk(LDCT_ROOT):

    if not any(p in root for p in ALL_PATIENTS):
        continue

    relative_path = os.path.relpath(root, LDCT_ROOT)
    seg_root = os.path.join(SEG_ROOT, relative_path)

    ldct_files = [f for f in files if f.endswith("_ldct.png")]
    ldct_files = sorted(ldct_files, key=lambda x: int(x.split('-')[1].split('_')[0]))

    for file in ldct_files:

        base_name = file.replace("_ldct.png", "")

        img_path  = os.path.join(root, base_name + "_ldct.png")
        lung_path = os.path.join(seg_root, base_name + "_lung_mask.png")
        bone_path = os.path.join(seg_root, base_name + "_bone_mask.png")
        soft_path = os.path.join(seg_root, base_name + "_soft_mask.png")

        if not all(os.path.exists(p) for p in [img_path, lung_path, bone_path, soft_path]):
            continue

        print(f"Processing {base_name}")

        img = load_image(img_path)

        lung_mask = (load_image(lung_path) > 0.5).astype(np.float32)
        bone_mask = (load_image(bone_path) > 0.5).astype(np.float32)
        soft_mask = (load_image(soft_path) > 0.5).astype(np.float32)

        # --------------------------
        # FIX 1: LIGHT lung expansion
        # --------------------------
        kernel = np.ones((3, 3), np.uint8)
        lung_mask = cv2.dilate(lung_mask, kernel, iterations=1)

        # --------------------------
        # SKIP SMALL LUNG SLICES
        # --------------------------
        if np.sum(lung_mask) / lung_mask.size < 0.05:
            continue

        # --------------------------
        # FIX 2: STRICT EXCLUSIVITY
        # --------------------------
        bone_mask = bone_mask * (1 - lung_mask)
        soft_mask = soft_mask * (1 - lung_mask)
        soft_mask = soft_mask * (1 - bone_mask)

        # Remaining region (safe)
        other_mask = 1 - (lung_mask + bone_mask + soft_mask)
        other_mask = np.clip(other_mask, 0, 1)

        # --------------------------
        # APPLY FILTERS
        # --------------------------
        I_bilateral = apply_bilateral(img)
        I_nlm = apply_nlm(img)
        I_bm3d = apply_bm3d(img)

        # --------------------------
        # FINAL FUSION (CORRECTED)
        # --------------------------
        I_region = (
            lung_mask * (0.6 * I_bilateral + 0.4 * I_nlm) +
            bone_mask * (0.7 * I_nlm + 0.3 * I_bm3d) +
            soft_mask * I_bm3d +
            other_mask * I_bm3d  
        )

        save_path = os.path.join(
            PHASE2_ROOT,
            relative_path,
            base_name + "_region.png"
        )

        save_image(I_region, save_path)

print("\nDONE: Region-wise denoising completed")
