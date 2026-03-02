import os
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means
from bm3d import bm3d

# ============================
# CONFIGURATION
# ============================

ENH_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"

MAX_PATIENTS = 8  # keep same control as Phase 1

# ============================
# Utility
# ============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)

# ============================
# Classical Filters (Full Image)
# ============================

def apply_bilateral(img):
    out = cv2.bilateralFilter(
        (img * 255).astype(np.uint8),
        d=7,               # weaker than Phase1
        sigmaColor=40,
        sigmaSpace=40
    )
    return out.astype(np.float32) / 255.0

def apply_nlm(img):
    out = denoise_nl_means(
        img,
        h=0.10,            # slightly stronger
        fast_mode=True,
        patch_size=5,
        patch_distance=6
    )
    return out.astype(np.float32)

def apply_bm3d(img):
    out = bm3d(img, sigma_psd=0.05)
    return out.astype(np.float32)

# ============================
# MAIN LOOP
# ============================

processed_patients = set()

for root, dirs, files in os.walk(ENH_ROOT):

    relative_path = os.path.relpath(root, ENH_ROOT)

    if relative_path == ".":
        continue

    # Identify patient ID from path
    patient_id = None
    for part in relative_path.split(os.sep):
        if part.startswith("LIDC-IDRI-"):
            patient_id = part
            break

    if patient_id is None:
        continue

    if patient_id not in processed_patients:
        if len(processed_patients) >= MAX_PATIENTS:
            break
        processed_patients.add(patient_id)

    for file in files:

        if not file.endswith("_enhanced.png"):
            continue

        base_name = file.replace("_enhanced.png", "")

        enhanced_path = os.path.join(root, base_name + "_enhanced.png")
        lung_path = os.path.join(root, base_name + "_lung_mask.png")
        bone_path = os.path.join(root, base_name + "_bone_mask.png")
        soft_path = os.path.join(root, base_name + "_soft_mask.png")

        if not all(os.path.exists(p) for p in [lung_path, bone_path, soft_path]):
            continue

        print(f"Processing {base_name}")

        # Load images
        img = load_image(enhanced_path)
        lung_mask = (load_image(lung_path) > 0.5).astype(np.float32)
        bone_mask = (load_image(bone_path) > 0.5).astype(np.float32)
        soft_mask = (load_image(soft_path) > 0.5).astype(np.float32)

        # Sanity: remove overlaps
        # Ensure exclusive masks
        bone_mask = bone_mask * (1 - lung_mask)
        soft_mask = soft_mask * (1 - lung_mask) * (1 - bone_mask)

        # Apply filters globally
        I_bilateral = apply_bilateral(img)
        I_nlm = apply_nlm(img)
        I_bm3d = apply_bm3d(img)

        # Region-Adaptive Blending
        mask_sum = lung_mask + bone_mask + soft_mask

        I_region = (
            lung_mask * I_bilateral +
            bone_mask * I_nlm +
            soft_mask * I_bm3d +
            (1 - mask_sum) * img
        )

        # Save output
        save_path = os.path.join(
            PHASE2_ROOT,
            relative_path,
            base_name + "_region_adaptive.png"
        )

        save_image(I_region, save_path)

print("Phase 2 Region-Adaptive Denoising completed.")