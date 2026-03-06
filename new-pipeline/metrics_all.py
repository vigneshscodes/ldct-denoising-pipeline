import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


# =========================================================
# ROOT PATHS
# =========================================================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
ENH_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"
PHASE1_ROOT = r"D:\CT_Datasets\Phase1_Classical"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"
CNN_ROOT = r"D:\CT_Datasets\Phase3_CNN_Refined"

OUTPUT_CSV = "all_metrics_summary.csv"


# =========================================================
# CNN SPLIT
# =========================================================

VALIDATION_PATIENTS = [
"LIDC-IDRI-0021","LIDC-IDRI-0022","LIDC-IDRI-0023"
]

TEST_PATIENTS = [
"LIDC-IDRI-0024","LIDC-IDRI-0025","LIDC-IDRI-0026"
]


# =========================================================
# UTILITIES
# =========================================================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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

    return structural_similarity(
        gt * mask,
        pred * mask,
        data_range=1.0
    )


# =========================================================
# STORAGE
# =========================================================

results = {
"LDCT": [],
"Enhanced": [],
"Bilateral_Lung": [],
"NLM_Lung": [],
"BM3D_Lung": [],
"Region_Adaptive": [],
"CNN_Refined": []
}

ssim_results = {k: [] for k in results.keys()}


# =========================================================
# MAIN LOOP
# =========================================================

for root, dirs, files in os.walk(LDCT_ROOT):

    for file in files:

        if not file.endswith("_ndct.png"):
            continue

        base = file.replace("_ndct.png","")

        ndct_path = os.path.join(root,file)
        ldct_path = os.path.join(root,base+"_ldct.png")

        relative = os.path.relpath(root,LDCT_ROOT)

        enh_folder = os.path.join(ENH_ROOT,relative)
        phase1_folder = os.path.join(PHASE1_ROOT,relative)
        phase2_folder = os.path.join(PHASE2_ROOT,relative)
        cnn_folder = os.path.join(CNN_ROOT,relative)

        enh_path = os.path.join(enh_folder,base+"_enhanced.png")
        mask_path = os.path.join(enh_folder,base+"_lung_mask.png")

        bil_path = os.path.join(phase1_folder,base+"_bilateral_lung.png")
        nlm_path = os.path.join(phase1_folder,base+"_nlm_lung.png")
        bm3d_path = os.path.join(phase1_folder,base+"_bm3d_lung.png")

        region_path = os.path.join(phase2_folder,base+"_region_adaptive.png")

        cnn_path = os.path.join(cnn_folder,base+"_cnn_refined.png")

        if not os.path.exists(mask_path):
            continue


        ndct = load_image(ndct_path)
        ldct = load_image(ldct_path)
        mask = (load_image(mask_path)>0.5).astype(np.uint8)

        if np.sum(mask)/mask.size < 0.05:
            continue


        # ---------------------------------------------------
        # LDCT
        # ---------------------------------------------------

        psnr = compute_psnr(ndct,ldct,mask)
        ssim = compute_ssim(ndct,ldct,mask)

        if psnr is not None:
            results["LDCT"].append(psnr)
            ssim_results["LDCT"].append(ssim)


        # ---------------------------------------------------
        # ENHANCED
        # ---------------------------------------------------

        if os.path.exists(enh_path):

            enh = load_image(enh_path)

            psnr = compute_psnr(ndct,enh,mask)
            ssim = compute_ssim(ndct,enh,mask)

            if psnr is not None:
                results["Enhanced"].append(psnr)
                ssim_results["Enhanced"].append(ssim)


        # ---------------------------------------------------
        # PHASE 1
        # ---------------------------------------------------

        for name,path in zip(
            ["Bilateral_Lung","NLM_Lung","BM3D_Lung"],
            [bil_path,nlm_path,bm3d_path]
        ):

            if os.path.exists(path):

                img = load_image(path)

                psnr = compute_psnr(ndct,img,mask)
                ssim = compute_ssim(ndct,img,mask)

                if psnr is not None:
                    results[name].append(psnr)
                    ssim_results[name].append(ssim)


        # ---------------------------------------------------
        # PHASE 2
        # ---------------------------------------------------

        if os.path.exists(region_path):

            region = load_image(region_path)

            psnr = compute_psnr(ndct,region,mask)
            ssim = compute_ssim(ndct,region,mask)

            if psnr is not None:
                results["Region_Adaptive"].append(psnr)
                ssim_results["Region_Adaptive"].append(ssim)


        # ---------------------------------------------------
        # CNN
        # ---------------------------------------------------

        patient_id = None
        for part in relative.split(os.sep):
            if part.startswith("LIDC-IDRI-"):
                patient_id = part
                break

        if patient_id in VALIDATION_PATIENTS + TEST_PATIENTS:

            if os.path.exists(cnn_path):

                cnn = load_image(cnn_path)

                psnr = compute_psnr(ndct,cnn,mask)
                ssim = compute_ssim(ndct,cnn,mask)

                if psnr is not None:
                    results["CNN_Refined"].append(psnr)
                    ssim_results["CNN_Refined"].append(ssim)


# =========================================================
# FINAL SUMMARY
# =========================================================

summary = []

print("\n==============================")
print("FINAL PIPELINE METRICS")
print("==============================\n")

for method in results.keys():

    if len(results[method])==0:
        continue

    psnr_mean = np.mean(results[method])
    psnr_std = np.std(results[method])

    ssim_mean = np.mean(ssim_results[method])
    ssim_std = np.std(ssim_results[method])

    print(method)
    print("PSNR:",psnr_mean,"±",psnr_std)
    print("SSIM:",ssim_mean,"±",ssim_std,"\n")

    summary.append({
        "Method":method,
        "PSNR_mean":psnr_mean,
        "PSNR_std":psnr_std,
        "SSIM_mean":ssim_mean,
        "SSIM_std":ssim_std
    })


df = pd.DataFrame(summary)
df.to_csv(OUTPUT_CSV,index=False)

print("Saved summary to",OUTPUT_CSV)