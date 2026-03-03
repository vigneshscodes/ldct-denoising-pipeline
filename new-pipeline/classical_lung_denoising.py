import os
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means
from bm3d import bm3d

# ----------------------------
# CONFIGURE PATHS
# ----------------------------
LDCT_ROOT = r"D:\CT_Datasets\LDCT"
ENH_ROOT  = r"D:\CT_Datasets\LDCT_Enhanced"
PHASE1_ROOT = r"D:\CT_Datasets\Phase1_Classical"

MAX_PATIENTS = 8   # ✅ LIMIT HERE
START_INDEX = 8    # skip first 8
END_INDEX   = 20   # stop at 20

# ----------------------------
# Utility Functions
# ----------------------------

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return img

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)

# ----------------------------
# Classical Filters
# ----------------------------

def apply_bilateral(img):
    bilateral = cv2.bilateralFilter(
        (img * 255).astype(np.uint8),
        d=9,
        sigmaColor=50,
        sigmaSpace=50
    )
    return bilateral.astype(np.float32) / 255.0

def apply_nlm(img):
    nlm = denoise_nl_means(
        img,
        h=0.08,
        fast_mode=True,
        patch_size=5,
        patch_distance=6
    )
    return nlm.astype(np.float32)

def apply_bm3d(img):
    bm3d_out = bm3d(img, sigma_psd=0.05)
    return bm3d_out.astype(np.float32)

# ----------------------------
# MAIN LOOP (PATIENT LIMITED)
# ----------------------------

processed_patients = set()

for root, dirs, files in os.walk(LDCT_ROOT):

    # Identify patient folder by relative path
    relative_path = os.path.relpath(root, LDCT_ROOT)

    if relative_path == ".":
        continue

    parts = relative_path.split(os.sep)
    patient_id = None

    for part in parts:
        if part.startswith("LIDC-IDRI-"):
            patient_id = part
            break
    if patient_id is None:
        continue

    if patient_id not in processed_patients:
        processed_patients.add(patient_id)

        patient_number = len(processed_patients)

        if patient_number <= START_INDEX:
            continue

        if patient_number > END_INDEX:
            break
        print("Now processing patient:", patient_id)

    for file in files:

        if not file.endswith("_ndct.png"):
            continue

        base_name = file.replace("_ndct.png", "")

        enh_folder = os.path.join(ENH_ROOT, relative_path)
        phase1_folder = os.path.join(PHASE1_ROOT, relative_path)

        enhanced_path = os.path.join(enh_folder, base_name + "_enhanced.png")
        mask_path     = os.path.join(enh_folder, base_name + "_lung_mask.png")

        if not os.path.exists(enhanced_path) or not os.path.exists(mask_path):
            continue

        img = load_image(enhanced_path)
        lung_mask = load_image(mask_path)
        lung_mask = (lung_mask > 0.5).astype(np.float32)

        if np.sum(lung_mask) / lung_mask.size < 0.05:
            continue

        print(f"Processing {base_name}")

        bilateral = apply_bilateral(img)
        nlm       = apply_nlm(img)
        bm3d_out  = apply_bm3d(img)

        bilateral_lung = lung_mask * bilateral + (1 - lung_mask) * img
        nlm_lung       = lung_mask * nlm + (1 - lung_mask) * img
        bm3d_lung      = lung_mask * bm3d_out + (1 - lung_mask) * img

        save_image(bilateral_lung, os.path.join(phase1_folder, base_name + "_bilateral_lung.png"))
        save_image(nlm_lung,       os.path.join(phase1_folder, base_name + "_nlm_lung.png"))
        save_image(bm3d_lung,      os.path.join(phase1_folder, base_name + "_bm3d_lung.png"))

print("Phase 1 classical lung-focused denoising completed.")