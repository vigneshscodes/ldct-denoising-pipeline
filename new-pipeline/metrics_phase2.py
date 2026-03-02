import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

# ============================
# ROOT PATHS
# ============================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
ENH_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"

OUTPUT_CSV = "phase2_region_adaptive_metrics.csv"
MAX_PATIENTS = 8

# ============================
# Utility
# ============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0

def compute_psnr(gt, pred, mask):
    gt_lung = gt[mask == 1]
    pred_lung = pred[mask == 1]

    if len(gt_lung) == 0:
        return None

    mse = np.mean((gt_lung - pred_lung) ** 2)
    if mse == 0:
        return 100.0

    return 10 * np.log10(1.0 / mse)

def compute_ssim(gt, pred, mask):
    return structural_similarity(gt * mask, pred * mask, data_range=1.0)

# ============================
# Collect first 8 patients
# ============================

patient_ids = set()

for root, dirs, files in os.walk(PHASE2_ROOT):
    for d in dirs:
        if d.startswith("LIDC-IDRI-"):
            patient_ids.add(d)

patient_ids = sorted(list(patient_ids))[:MAX_PATIENTS]

print("\nEvaluating Phase 2 STRICT patients:")
print(patient_ids)

# ============================
# Storage
# ============================

results = {
    "LDCT": [],
    "Enhanced": [],
    "Region_Adaptive": []
}

ssim_results = {k: [] for k in results.keys()}

# ============================
# MAIN LOOP
# ============================

for root, dirs, files in os.walk(PHASE2_ROOT):

    for file in files:

        if not file.endswith("_region_adaptive.png"):
            continue

        # Identify patient ID from path
        patient_id = None
        for pid in patient_ids:
            if pid in root:
                patient_id = pid
                break

        if patient_id is None:
            continue

        base_name = file.replace("_region_adaptive.png", "")
        relative_path = os.path.relpath(root, PHASE2_ROOT)

        # Mirror paths
        ldct_folder = os.path.join(LDCT_ROOT, relative_path)
        enh_folder = os.path.join(ENH_ROOT, relative_path)

        ndct_path = os.path.join(ldct_folder, base_name + "_ndct.png")
        ldct_path = os.path.join(ldct_folder, base_name + "_ldct.png")
        enh_path = os.path.join(enh_folder, base_name + "_enhanced.png")
        mask_path = os.path.join(enh_folder, base_name + "_lung_mask.png")
        region_path = os.path.join(root, base_name + "_region_adaptive.png")

        if not all(os.path.exists(p) for p in 
                   [ndct_path, ldct_path, enh_path, mask_path, region_path]):
            continue

        ndct = load_image(ndct_path)
        ldct = load_image(ldct_path)
        enh = load_image(enh_path)
        region = load_image(region_path)
        mask = (load_image(mask_path) > 0.5).astype(np.uint8)

        if np.sum(mask) / mask.size < 0.05:
            continue

        for name, img in zip(
            ["LDCT", "Enhanced", "Region_Adaptive"],
            [ldct, enh, region]
        ):
            psnr = compute_psnr(ndct, img, mask)
            ssim = compute_ssim(ndct, img, mask)

            if psnr is not None:
                results[name].append(psnr)
                ssim_results[name].append(ssim)

# ============================
# FINAL RESULTS
# ============================

print("\n===== Phase 2 Region-Adaptive Results =====\n")

for method in results.keys():
    if len(results[method]) == 0:
        print(f"{method}: No valid slices\n")
        continue

    print(method)
    print(f"  PSNR: {np.mean(results[method]):.3f} ± {np.std(results[method]):.3f}")
    print(f"  SSIM: {np.mean(ssim_results[method]):.4f} ± {np.std(ssim_results[method]):.4f}")
    print()

df = pd.DataFrame({
    f"{k}_PSNR": results[k] for k in results.keys()
} | {
    f"{k}_SSIM": ssim_results[k] for k in results.keys()
})

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved results to {OUTPUT_CSV}")