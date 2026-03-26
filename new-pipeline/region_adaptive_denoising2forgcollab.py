# ============================
# REGION-WISE DENOISING (FINAL 100/100)
# ============================

import os
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means
from bm3d import bm3d

# ============================
# CONFIGURATION
# ============================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
SEG_ROOT  = r"D:\CT_Datasets\Segmentation"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_Output"

EVAL_PATIENT = "LIDC-IDRI-0001"

os.makedirs(PHASE2_ROOT, exist_ok=True)


# ============================
# Utility
# ============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)


# ============================
# Classical Filters (TUNED)
# ============================

def apply_bilateral(img):
    #  Stronger for lung
    out = cv2.bilateralFilter(
        (img * 255).astype(np.uint8),
        d=9,
        sigmaColor=60,
        sigmaSpace=60
    )
    return out.astype(np.float32) / 255.0


def apply_nlm(img):
    return denoise_nl_means(
        img,
        h=0.12,   # slightly stronger
        fast_mode=True,
        patch_size=5,
        patch_distance=6
    ).astype(np.float32)


def apply_bm3d(img):
    #  Stronger BM3D (key improvement)
    return bm3d(img, sigma_psd=0.05).astype(np.float32)


# ============================
# MAIN LOOP
# ============================

for root, dirs, files in os.walk(LDCT_ROOT):

    if EVAL_PATIENT not in root:
        continue

    relative_path = os.path.relpath(root, LDCT_ROOT)
    seg_root = os.path.join(SEG_ROOT, relative_path)

    #  SORT FILES (correct order)
    ldct_files = [f for f in files if f.endswith("_ldct.png")]
    ldct_files = sorted(
        ldct_files,
        key=lambda x: int(x.split('-')[1].split('_')[0])
    )

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
        lung_mask = load_image(lung_path)
        bone_mask = load_image(bone_path)
        soft_mask = load_image(soft_path)

        if img is None or lung_mask is None:
            continue

        # --------------------------
        # BINARIZE MASKS
        # --------------------------
        lung_mask = (lung_mask > 0.5).astype(np.float32)
        bone_mask = (bone_mask > 0.5).astype(np.float32)
        soft_mask = (soft_mask > 0.5).astype(np.float32)

        # --------------------------
        # SKIP NON-LUNG SLICES
        # --------------------------
        if np.sum(lung_mask) / lung_mask.size < 0.05:
            continue

        # --------------------------
        # ENSURE EXCLUSIVE MASKS
        # --------------------------
        bone_mask = bone_mask * (1 - lung_mask)
        soft_mask = soft_mask * (1 - lung_mask) * (1 - bone_mask)

        # --------------------------
        # APPLY FILTERS
        # --------------------------
        I_bilateral = apply_bilateral(img)
        I_nlm = apply_nlm(img)

        #  BM3D optimization (keep)
        if np.sum(soft_mask) / soft_mask.size < 0.02:
            I_bm3d = img
        else:
            I_bm3d = apply_bm3d(img)

        mask_sum = lung_mask + bone_mask + soft_mask

        # --------------------------
        #  IMPROVED REGION FUSION (CRITICAL)
        # --------------------------
        I_region = (
            lung_mask * (0.7 * I_bilateral + 0.3 * img) +
            bone_mask * I_nlm +
            soft_mask * (0.8 * I_bm3d + 0.2 * img) +
            (1 - mask_sum) * img
        )

        save_path = os.path.join(
            PHASE2_ROOT,
            relative_path,
            base_name + "_region.png"
        )

        save_image(I_region, save_path)

print("\nDone: Region-wise denoising completed")