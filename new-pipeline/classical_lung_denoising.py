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

# Evaluation patients (6 new ones)
EVAL_PATIENTS = [
"LIDC-IDRI-0021",
"LIDC-IDRI-0022",
"LIDC-IDRI-0023",
"LIDC-IDRI-0024",
"LIDC-IDRI-0025",
"LIDC-IDRI-0026"
]

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
# MAIN PROCESSING LOOP
# ----------------------------

for root, dirs, files in os.walk(LDCT_ROOT):

    relative_path = os.path.relpath(root, LDCT_ROOT)
    if relative_path == ".":
        continue

    # Detect patient ID from path
    patient_id = None
    for part in root.split(os.sep):
        if part.startswith("LIDC-IDRI-"):
            patient_id = part
            break

    # Process ONLY evaluation patients
    if patient_id is None or patient_id not in EVAL_PATIENTS:
        continue


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


        # Skip slices with very small lung area
        if np.sum(lung_mask) / lung_mask.size < 0.05:
            continue


        print(f"Processing {patient_id} → {base_name}")


        # Apply classical filters
        bilateral = apply_bilateral(img)
        nlm       = apply_nlm(img)
        bm3d_out  = apply_bm3d(img)


        # Apply filters only inside lung
        bilateral_lung = lung_mask * bilateral + (1 - lung_mask) * img
        nlm_lung       = lung_mask * nlm + (1 - lung_mask) * img
        bm3d_lung      = lung_mask * bm3d_out + (1 - lung_mask) * img


        # Save outputs
        save_image(bilateral_lung, os.path.join(phase1_folder, base_name + "_bilateral_lung.png"))
        save_image(nlm_lung,       os.path.join(phase1_folder, base_name + "_nlm_lung.png"))
        save_image(bm3d_lung,      os.path.join(phase1_folder, base_name + "_bm3d_lung.png"))


print("\nPhase 1 classical lung-focused denoising completed.")