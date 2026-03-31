import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim


# =========================================================
# ROOT PATHS (FINAL CORRECT)
# =========================================================

LDCT_ROOT   = r"D:\CT_Datasets\LDCT"
SEG_ROOT    = r"D:\CT_Datasets\Segmentation"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_Output"
CNN_ROOT    = r"D:\CT_Datasets\Phase3_CNN_Refined"

OUTPUT_CSV = "final_metrics_summary.csv"


# =========================================================
# PATIENT SPLIT
# =========================================================

VAL_PATIENTS = [f"LIDC-IDRI-{i:04d}" for i in range(21, 24)]
TEST_PATIENTS = [f"LIDC-IDRI-{i:04d}" for i in range(24, 27)]


# =========================================================
# UTILITIES
# =========================================================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


def compute_psnr(gt, pred, mask):

    gt = gt[mask == 1]
    pred = pred[mask == 1]

    if len(gt) == 0:
        return None

    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 100

    return 10 * np.log10(1.0 / mse)


def compute_ssim(gt, pred, mask):
    return ssim(gt * mask, pred * mask, data_range=1.0)


# =========================================================
# STORAGE
# =========================================================

results = {
    "LDCT": [],
    "REGION": [],
    "CNN_VAL": [],
    "CNN_TEST": []
}

ssim_results = {k: [] for k in results.keys()}


# =========================================================
# MAIN LOOP
# =========================================================

for root, dirs, files in os.walk(LDCT_ROOT):

    for file in files:

        if not file.endswith("_ndct.png"):
            continue

        base = file.replace("_ndct.png", "")
        relative = os.path.relpath(root, LDCT_ROOT)

        # Paths
        ndct_path = os.path.join(root, file)
        ldct_path = os.path.join(root, base + "_ldct.png")

        seg_path   = os.path.join(SEG_ROOT, relative, base + "_lung_mask.png")
        region_path = os.path.join(PHASE2_ROOT, relative, base + "_region.png")
        cnn_path    = os.path.join(CNN_ROOT, relative, base + "_cnn_refined.png")

        if not os.path.exists(seg_path):
            continue

        # Load
        ndct = load_image(ndct_path)
        ldct = load_image(ldct_path)
        mask = load_image(seg_path)

        if ndct is None or ldct is None or mask is None:
            continue

        mask = (mask > 0.5).astype(np.uint8)

        # 🔥 SAME FIX as your best pipeline
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        # Skip small lung
        if np.sum(mask) / mask.size < 0.05:
            continue

        # ---------------------------------------------------
        # LDCT
        # ---------------------------------------------------
        ps = compute_psnr(ndct, ldct, mask)
        ss = compute_ssim(ndct, ldct, mask)

        if ps is not None:
            results["LDCT"].append(ps)
            ssim_results["LDCT"].append(ss)

        # ---------------------------------------------------
        # REGION
        # ---------------------------------------------------
        if os.path.exists(region_path):

            region = load_image(region_path)

            if region is not None:
                ps = compute_psnr(ndct, region, mask)
                ss = compute_ssim(ndct, region, mask)

                if ps is not None:
                    results["REGION"].append(ps)
                    ssim_results["REGION"].append(ss)

        # ---------------------------------------------------
        # CNN
        # ---------------------------------------------------
        if os.path.exists(cnn_path):

            cnn = load_image(cnn_path)

            if cnn is None:
                continue

            ps = compute_psnr(ndct, cnn, mask)
            ss = compute_ssim(ndct, cnn, mask)

            if ps is None:
                continue

            # detect patient
            patient_id = None
            for part in relative.split(os.sep):
                if part.startswith("LIDC-IDRI-"):
                    patient_id = part
                    break

            if patient_id in VAL_PATIENTS:
                results["CNN_VAL"].append(ps)
                ssim_results["CNN_VAL"].append(ss)

            elif patient_id in TEST_PATIENTS:
                results["CNN_TEST"].append(ps)
                ssim_results["CNN_TEST"].append(ss)


# =========================================================
# FINAL SUMMARY
# =========================================================

summary = []

print("\n==============================")
print("FINAL RESULTS (LUNG REGION)")
print("==============================\n")

for method in results.keys():

    if len(results[method]) == 0:
        continue

    psnr_mean = np.mean(results[method])
    psnr_std  = np.std(results[method])

    ssim_mean = np.mean(ssim_results[method])
    ssim_std  = np.std(ssim_results[method])

    print(method)
    print(f"PSNR: {psnr_mean:.2f} ± {psnr_std:.2f}")
    print(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}\n")

    summary.append({
        "Method": method,
        "PSNR_mean": psnr_mean,
        "PSNR_std": psnr_std,
        "SSIM_mean": ssim_mean,
        "SSIM_std": ssim_std
    })


df = pd.DataFrame(summary)
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Saved summary to:", OUTPUT_CSV)