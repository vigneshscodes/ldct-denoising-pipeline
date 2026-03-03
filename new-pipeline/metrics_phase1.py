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
PHASE1_ROOT = r"D:\CT_Datasets\Phase1_Classical"

OUTPUT_CSV = "phase1_comparison_20patients.csv"
MAX_PATIENTS = 20

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

for root, dirs, files in os.walk(PHASE1_ROOT):
    for d in dirs:
        if d.startswith("LIDC-IDRI-"):
            patient_ids.add(d)

patient_ids = sorted(list(patient_ids))[:MAX_PATIENTS]

print("\nEvaluating STRICT patients:")
print(patient_ids)

# ============================
# Storage
# ============================

results = {
    "LDCT": [],
    "Enhanced": [],
    "Bilateral_Lung": [],
    "NLM_Lung": [],
    "BM3D_Lung": []
}

ssim_results = {k: [] for k in results.keys()}

# ============================
# MAIN EVALUATION
# ============================

for root, dirs, files in os.walk(PHASE1_ROOT):

    for file in files:

        if not file.endswith("_bilateral_lung.png"):
            continue

        # Identify patient ID from path
        patient_id = None
        for pid in patient_ids:
            if pid in root:
                patient_id = pid
                break

        if patient_id is None:
            continue  # Skip if not in selected 8 patients

        base_name = file.replace("_bilateral_lung.png", "")

        # Build corresponding paths (mirror structure)
        relative_path = os.path.relpath(root, PHASE1_ROOT)

        ldct_folder = os.path.join(LDCT_ROOT, relative_path)
        enh_folder = os.path.join(ENH_ROOT, relative_path)

        ndct_path = os.path.join(ldct_folder, base_name + "_ndct.png")
        ldct_path = os.path.join(ldct_folder, base_name + "_ldct.png")
        enh_path  = os.path.join(enh_folder, base_name + "_enhanced.png")
        mask_path = os.path.join(enh_folder, base_name + "_lung_mask.png")

        bil_path  = os.path.join(root, base_name + "_bilateral_lung.png")
        nlm_path  = os.path.join(root, base_name + "_nlm_lung.png")
        bm3d_path = os.path.join(root, base_name + "_bm3d_lung.png")

        if not all(os.path.exists(p) for p in 
                   [ndct_path, ldct_path, enh_path, mask_path, bil_path, nlm_path, bm3d_path]):
            continue

        ndct = load_image(ndct_path)
        ldct = load_image(ldct_path)
        enh  = load_image(enh_path)
        bil  = load_image(bil_path)
        nlm  = load_image(nlm_path)
        bm3d_img = load_image(bm3d_path)
        mask = (load_image(mask_path) > 0.5).astype(np.uint8)

        # Skip small lung slices
        if np.sum(mask) / mask.size < 0.05:
            continue

        for name, img in zip(
            ["LDCT", "Enhanced", "Bilateral_Lung", "NLM_Lung", "BM3D_Lung"],
            [ldct, enh, bil, nlm, bm3d_img]
        ):
            psnr = compute_psnr(ndct, img, mask)
            ssim = compute_ssim(ndct, img, mask)

            if psnr is not None:
                results[name].append(psnr)
                ssim_results[name].append(ssim)

# ============================
# FINAL RESULTS
# ============================

print("\n===== Phase 1 =====\n")

for method in results.keys():
    if len(results[method]) == 0:
        print(f"{method}: No valid slices found\n")
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