# main_pipeline.py

import os
import numpy as np
import cv2

from preprocessing import load_dicom, apply_lung_window
from ldct_simulation import simulate_ldct
from enhancement import enhance_ldct
from segmentation import (
    load_segmentation_model,
    predict_lung_mask,
    postprocess_lung_mask,
    create_bone_mask,
    create_soft_tissue_mask
)

# --------------------------
# CONFIGURE PATHS HERE
# --------------------------
INPUT_ROOT  = r"D:\CT_Datasets\NDCT"
LDCT_ROOT   = r"D:\CT_Datasets\LDCT"
ENH_ROOT    = r"D:\CT_Datasets\LDCT_Enhanced"

os.makedirs(LDCT_ROOT, exist_ok=True)
os.makedirs(ENH_ROOT, exist_ok=True)


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
    img_uint8 = (img_norm * 255).astype(np.uint8)
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
for root, dirs, files in os.walk(INPUT_ROOT):
    for file in files:
        if not file.lower().endswith(".dcm"):
            continue

        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, INPUT_ROOT)

        ldct_folder = os.path.join(LDCT_ROOT, relative_path)
        enh_folder  = os.path.join(ENH_ROOT, relative_path)

        os.makedirs(ldct_folder, exist_ok=True)
        os.makedirs(enh_folder, exist_ok=True)

        ldct_path = os.path.join(ldct_folder, file)
        enh_path  = os.path.join(enh_folder, file)

        base_name = os.path.splitext(file)[0]

        # --------------------------
        # STEP 1: Load & Window
        # --------------------------
        ds, img_hu = load_dicom(input_path)
        img_norm, lower, upper = apply_lung_window(img_hu)

        # --------------------------
        # STEP 2: Simulate LDCT
        # --------------------------
        ldct_norm, ldct_hu = simulate_ldct(img_norm, lower, upper)
        save_dicom(ds, ldct_hu, ldct_path)

        # --------------------------
        # STEP 3: Enhancement
        # --------------------------
        enhanced_norm = enhance_ldct(ldct_norm)
        enhanced_hu = enhanced_norm * (upper - lower) + lower
        save_dicom(ds, enhanced_hu, enh_path)

        # --------------------------
        # STEP 4: Lung Segmentation
        # --------------------------
        lung_prob = predict_lung_mask(seg_model, enhanced_norm)
        lung_mask = postprocess_lung_mask(lung_prob)

        # Quick sanity check (only middle slices will show larger %)
        lung_percent = np.sum(lung_mask) / lung_mask.size * 100
        print(f"{file} → Lung area: {lung_percent:.2f}%")

        # Overlay for visual verification
        overlay = enhanced_norm.copy()
        overlay[lung_mask == 1] = 1

        # --------------------------
        # STEP 5: Bone Mask (HU > 300)
        # --------------------------
        bone_mask = create_bone_mask(enhanced_hu)

        # --------------------------
        # STEP 6: Body Mask
        # --------------------------
        body_mask = (enhanced_hu > -500).astype(np.uint8)

        # --------------------------
        # STEP 7: Soft Tissue Mask
        # --------------------------
        soft_mask = create_soft_tissue_mask(body_mask, lung_mask, bone_mask)

        # --------------------------
        # SAVE VERIFICATION OUTPUTS
        # --------------------------
        save_png(lung_mask, os.path.join(enh_folder, f"{base_name}_lung_mask.png"))
        save_png(bone_mask, os.path.join(enh_folder, f"{base_name}_bone_mask.png"))
        save_png(soft_mask, os.path.join(enh_folder, f"{base_name}_soft_mask.png"))
        save_png(overlay, os.path.join(enh_folder, f"{base_name}_lung_overlay.png"))

        # Optional debug images
        save_png(img_norm, os.path.join(ldct_folder, f"{base_name}_ndct.png"))
        save_png(ldct_norm, os.path.join(ldct_folder, f"{base_name}_ldct.png"))
        save_png(enhanced_norm, os.path.join(enh_folder, f"{base_name}_enhanced.png"))

print("Pipeline completed successfully.")