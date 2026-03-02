import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
ENH_ROOT  = r"D:\CT_Datasets\LDCT_Enhanced"
PHASE1_ROOT = r"D:\CT_Datasets\Phase1_Classical"

OUTPUT_CSV = "phase1_comparison_8patients.csv"
MAX_PATIENTS = 8

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

results = {
    "LDCT": [],
    "Enhanced": [],
    "Bilateral_Lung": [],
    "NLM_Lung": [],
    "BM3D_Lung": []
}

ssim_results = {k: [] for k in results.keys()}

# 🔒 STRICT: Only look at Phase1 folders
patient_folders = sorted([
    f for f in os.listdir(PHASE1_ROOT)
    if f.startswith("LIDC-IDRI-")
])[:MAX_PATIENTS]

print("Evaluating ONLY these patients:", patient_folders)

print("Evaluating ONLY these patients:")
print(patient_folders)

for patient_id in patient_folders:

    patient_ldct_path = os.path.join(LDCT_ROOT, patient_id)
    patient_enh_path  = os.path.join(ENH_ROOT, patient_id)
    patient_phase1_path = os.path.join(PHASE1_ROOT, patient_id)

    if not os.path.isdir(patient_phase1_path):
        continue

    for file in os.listdir(patient_phase1_path):

        if not file.endswith("_bilateral_lung.png"):
            continue

        base_name = file.replace("_bilateral_lung.png", "")

        ndct_path = os.path.join(patient_ldct_path, base_name + "_ndct.png")
        ldct_path = os.path.join(patient_ldct_path, base_name + "_ldct.png")
        enh_path  = os.path.join(patient_enh_path, base_name + "_enhanced.png")
        mask_path = os.path.join(patient_enh_path, base_name + "_lung_mask.png")

        bil_path  = os.path.join(patient_phase1_path, base_name + "_bilateral_lung.png")
        nlm_path  = os.path.join(patient_phase1_path, base_name + "_nlm_lung.png")
        bm3d_path = os.path.join(patient_phase1_path, base_name + "_bm3d_lung.png")

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

print("\n===== Phase 1 (STRICT 8 Patients) =====\n")

for method in results.keys():
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