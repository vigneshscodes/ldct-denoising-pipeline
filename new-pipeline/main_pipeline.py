# main_pipeline.py

import os
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt

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

def verify_lung_presence(img_hu):

    # Count voxels near air HU (-1000)
    air_pixels = np.sum(img_hu < -900)
    total_pixels = img_hu.size

    air_percent = air_pixels / total_pixels * 100

    return air_percent

# --------------------------
# PROCESS LOOP
# --------------------------

for root, dirs, files in os.walk(INPUT_ROOT):
    for file in files:
        if file.lower().endswith(".dcm"):

            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, INPUT_ROOT)

            ldct_folder = os.path.join(LDCT_ROOT, relative_path)
            enh_folder  = os.path.join(ENH_ROOT, relative_path)

            os.makedirs(ldct_folder, exist_ok=True)
            os.makedirs(enh_folder, exist_ok=True)

            ldct_path = os.path.join(ldct_folder, file)
            enh_path  = os.path.join(enh_folder, file)

            # ---- STEP 1: Load & Window ----
            ds, img_hu = load_dicom(input_path)

            # --- Verify if slice contains lung ---
            air_percent = verify_lung_presence(img_hu)
            print(f"Air percentage: {air_percent:.2f}%")

            if air_percent < 5:
                print("WARNING: This slice likely does NOT contain lungs.")
            else:
                print("Slice likely contains lung region.")

            img_norm, lower, upper = apply_lung_window(img_hu)

            # ---- STEP 2: Simulate LDCT ----
            ldct_norm, ldct_hu = simulate_ldct(img_norm, lower, upper)

            # Save LDCT DICOM
            save_dicom(ds, ldct_hu, ldct_path)

            # ---- STEP 3: Enhancement ----
            enhanced_norm = enhance_ldct(ldct_norm)
            enhanced_hu = enhanced_norm * (upper - lower) + lower

            save_dicom(ds, enhanced_hu, enh_path)

            # ---- STEP 4: Lung Segmentation ----
            lung_prob = predict_lung_mask(seg_model, enhanced_norm)
            lung_mask = postprocess_lung_mask(lung_prob)

            # ---- Segmentation Verification ----
            lung_percent = np.sum(lung_mask) / lung_mask.size * 100
            print(f"Lung mask percentage: {lung_percent:.2f}%")

            # Overlay for visual inspection
            overlay = enhanced_norm.copy()
            overlay[lung_mask == 1] = 1
            save_png(overlay, os.path.join(enh_folder, "lung_overlay.png"))

            # ---- STEP 5: Bone Mask (HU-based) ----
            bone_mask = create_bone_mask(enhanced_hu)

            # ---- STEP 6: Body Mask (Basic Threshold) ----
            body_mask = (enhanced_hu > -500).astype(np.uint8)

            # ---- STEP 7: Soft Tissue Mask ----
            soft_mask = create_soft_tissue_mask(body_mask, lung_mask, bone_mask)

            # Save masks for verification
            save_png(lung_mask, os.path.join(enh_folder, "lung_mask.png"))
            save_png(bone_mask, os.path.join(enh_folder, "bone_mask.png"))
            save_png(soft_mask, os.path.join(enh_folder, "soft_mask.png"))

            # ---- OPTIONAL: Save verification PNG ----
            save_png(img_norm, os.path.join(ldct_folder, "ndct_debug.png"))
            save_png(ldct_norm, os.path.join(ldct_folder, "ldct_debug.png"))
            save_png(enhanced_norm, os.path.join(enh_folder, "enhanced_debug.png"))

print("Pipeline completed successfully.")
