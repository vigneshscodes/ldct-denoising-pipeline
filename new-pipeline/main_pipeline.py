import os
import numpy as np
import cv2

from preprocessing import load_dicom, apply_lung_window
from ldct_simulation import simulate_ldct
from segmentation import (
    load_segmentation_model,
    predict_lung_mask,
    create_bone_mask,
    create_soft_tissue_mask
)

# --------------------------
# CONFIGURE PATHS
# --------------------------
INPUT_ROOTS = [
    r"D:\CT_Datasets\NDCT",
    r"D:\CT_Datasets\NDCT_Eval"
]

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
SEG_ROOT  = r"D:\CT_Datasets\Segmentation"

os.makedirs(LDCT_ROOT, exist_ok=True)
os.makedirs(SEG_ROOT, exist_ok=True)

# --------------------------
# PATIENT SPLITS
# --------------------------
TRAIN_PATIENTS = [
"LIDC-IDRI-0001","LIDC-IDRI-0002","LIDC-IDRI-0003","LIDC-IDRI-0004","LIDC-IDRI-0005",
"LIDC-IDRI-0006","LIDC-IDRI-0007","LIDC-IDRI-0008","LIDC-IDRI-0009","LIDC-IDRI-0010",
"LIDC-IDRI-0011","LIDC-IDRI-0012","LIDC-IDRI-0013","LIDC-IDRI-0014","LIDC-IDRI-0015",
"LIDC-IDRI-0016","LIDC-IDRI-0017","LIDC-IDRI-0018","LIDC-IDRI-0019","LIDC-IDRI-0020"
]

VAL_PATIENTS = [
"LIDC-IDRI-0021","LIDC-IDRI-0022","LIDC-IDRI-0023"
]

TEST_PATIENTS = [
"LIDC-IDRI-0024","LIDC-IDRI-0025","LIDC-IDRI-0026"
]

ALL_PATIENTS = TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS

# --------------------------
# Utility Functions
# --------------------------
def save_dicom(original_ds, new_hu_img, save_path):
    ds_copy = original_ds.copy()

    slope = getattr(ds_copy, "RescaleSlope", 1)
    intercept = getattr(ds_copy, "RescaleIntercept", 0)

    pixel_data = (new_hu_img - intercept) / slope
    pixel_data = pixel_data.astype(ds_copy.pixel_array.dtype)

    ds_copy.PixelData = pixel_data.tobytes()
    ds_copy.save_as(save_path)


def save_png(img_norm, save_path):
    img_uint8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(save_path, img_uint8)


# --------------------------
# Load Segmentation Model
# --------------------------
MODEL_PATH = "model_100_epoch.pth"
seg_model = load_segmentation_model(MODEL_PATH)

print("Segmentation model loaded successfully.")


# --------------------------
# PROCESS LOOP
# --------------------------
for INPUT_ROOT in INPUT_ROOTS:
    for root, dirs, files in os.walk(INPUT_ROOT):

        # Detect patient ID
        patient_id = None
        for part in root.split(os.sep):
            if part.startswith("LIDC-IDRI-"):
                patient_id = part
                break

        if patient_id is None or patient_id not in ALL_PATIENTS:
            continue

        # Assign split
        if patient_id in TRAIN_PATIENTS:
            split = "train"
        elif patient_id in VAL_PATIENTS:
            split = "val"
        elif patient_id in TEST_PATIENTS:
            split = "test"
        else:
            continue

        for file in files:

            if not file.lower().endswith(".dcm"):
                continue

            input_path = os.path.join(root, file)

            relative_subpath = os.path.relpath(root, INPUT_ROOT)
            relative_path = os.path.join(split, relative_subpath)

            ldct_folder = os.path.join(LDCT_ROOT, relative_path)
            seg_folder  = os.path.join(SEG_ROOT, relative_path)

            os.makedirs(ldct_folder, exist_ok=True)
            os.makedirs(seg_folder, exist_ok=True)

            ldct_path = os.path.join(ldct_folder, file)
            base_name = os.path.splitext(file)[0]

            # --------------------------
            # STEP 1: Load NDCT
            # --------------------------
            ds, img_hu = load_dicom(input_path)

            # --------------------------
            # STEP 2: Region masks (from NDCT)
            # --------------------------
            bone_mask = create_bone_mask(img_hu)
            body_mask = (img_hu > -800).astype(np.uint8)

            # --------------------------
            # STEP 3: Lung window
            # --------------------------
            img_hu, img_norm, lower, upper = apply_lung_window(img_hu)

            # --------------------------
            # STEP 4: Simulate LDCT (FIXED)
            # --------------------------
            ldct_norm, ldct_hu = simulate_ldct(img_hu)

            save_dicom(ds, ldct_hu, ldct_path)

            # Save NDCT + LDCT PNGs
            save_png(img_norm, os.path.join(ldct_folder, f"{base_name}_ndct.png"))
            save_png(ldct_norm, os.path.join(ldct_folder, f"{base_name}_ldct.png"))

            # --------------------------
            # STEP 5: Segmentation (UPDATED)
            # --------------------------
            lung_prob, lung_mask = predict_lung_mask(seg_model, ldct_norm)

            lung_percent = np.sum(lung_mask) / lung_mask.size * 100
            print(f"{split} | {file} → Lung area: {lung_percent:.2f}%")

            # --------------------------
            # STEP 6: Soft tissue mask
            # --------------------------
            soft_mask = create_soft_tissue_mask(body_mask, lung_mask, bone_mask)

            # --------------------------
            # SAVE SEGMENTATION OUTPUTS
            # --------------------------
            save_png(lung_mask, os.path.join(seg_folder, f"{base_name}_lung_mask.png"))
            save_png(bone_mask, os.path.join(seg_folder, f"{base_name}_bone_mask.png"))
            save_png(soft_mask, os.path.join(seg_folder, f"{base_name}_soft_mask.png"))

print("Pipeline completed successfully.")