# ============================
# ENHANCEMENT (1 PATIENT ONLY)
# ============================

import os
import cv2
import numpy as np

# ============================
# CONFIGURATION
# ============================

PHASE2_ROOT = r"D:\CT_Datasets\Phase2_Output"
ENH_ROOT    = r"D:\CT_Datasets\Final_Enhanced"

EVAL_PATIENT = "LIDC-IDRI-0001"

os.makedirs(ENH_ROOT, exist_ok=True)

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
# CLAHE (16-bit safe)
# ============================

def apply_clahe(img):

    img_uint16 = (img * 65535).astype(np.uint16)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    out = clahe.apply(img_uint16)
    out = out.astype(np.float32) / 65535.0

    return out

# ============================
# Mild Sharpening
# ============================

def mild_sharpen(img):

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharpened = img + 0.25 * (img - blurred)

    return np.clip(sharpened, 0, 1)

# ============================
# MAIN LOOP
# ============================

for root, dirs, files in os.walk(PHASE2_ROOT):

    if EVAL_PATIENT not in root:
        continue

    relative_path = os.path.relpath(root, PHASE2_ROOT)
    out_root = os.path.join(ENH_ROOT, relative_path)

    # sort properly
    region_files = [f for f in files if f.endswith("_region.png")]
    region_files = sorted(
        region_files,
        key=lambda x: int(x.split('-')[1].split('_')[0])
    )

    for file in region_files:

        base_name = file.replace("_region.png", "")

        img_path = os.path.join(root, file)

        print(f"Enhancing {base_name}")

        img = load_image(img_path)

        if img is None:
            continue

        # --------------------------
        # Enhancement
        # --------------------------
        img_enh = apply_clahe(img)
        img_enh = mild_sharpen(img_enh)

        # --------------------------
        # Save
        # --------------------------
        save_path = os.path.join(
            out_root,
            base_name + "_enhanced.png"
        )

        save_image(img_enh, save_path)

print("\nDONE: Enhancement completed for patient")