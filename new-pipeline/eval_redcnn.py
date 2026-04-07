import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# =============================
# PATHS
# =============================
LDCT_ROOT   = r"D:\CT_Datasets\LDCT"
SEG_ROOT    = r"D:\CT_Datasets\Segmentation"
REDCNN_ROOT = r"D:\RED_CNN_Output"

# =============================
# BUILD RED-CNN INDEX (PATIENT + SLICE)
# =============================
redcnn_index = {}

for root, _, files in os.walk(REDCNN_ROOT):
    for f in files:
        if f.endswith("_redcnn.png"):

            # extract slice id
            name = f.replace("_redcnn.png", "")
            slice_id = name.split("_")[-1]

            # extract patient id
            parts = root.split(os.sep)
            patient_id = None
            for p in parts:
                if p.startswith("LIDC-IDRI-"):
                    patient_id = p
                    break

            if patient_id is not None:
                key = f"{patient_id}_{slice_id}"
                redcnn_index[key] = os.path.join(root, f)

print(f"Indexed {len(redcnn_index)} slices")

# =============================
# UTILS
# =============================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0 if img is not None else None

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

# =============================
# MAIN LOOP
# =============================
psnr_list = []
ssim_list = []
missing = 0

for root, _, files in os.walk(LDCT_ROOT):

    for file in files:

        if not file.endswith("_ndct.png"):
            continue

        base = file.replace("_ndct.png", "")
        slice_id = base.split("_")[-1]

        # extract patient id
        patient_id = None
        for part in root.split(os.sep):
            if part.startswith("LIDC-IDRI-"):
                patient_id = part
                break

        if patient_id is None:
            continue

        key = f"{patient_id}_{slice_id}"

        ndct_path = os.path.join(root, file)

        mask_path = os.path.join(
            SEG_ROOT,
            os.path.relpath(root, LDCT_ROOT),
            base + "_lung_mask.png"
        )

        redcnn_path = redcnn_index.get(key, None)

        if redcnn_path is None or not os.path.exists(mask_path):
            missing += 1
            continue

        ndct = load_image(ndct_path)
        pred = load_image(redcnn_path)
        mask = load_image(mask_path)

        if ndct is None or pred is None or mask is None:
            continue

        # resize pred → match NDCT
        pred = cv2.resize(pred, (ndct.shape[1], ndct.shape[0]))

        mask = (mask > 0.5).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        if np.sum(mask) / mask.size < 0.05:
            continue

        ps = compute_psnr(ndct, pred, mask)
        ss = compute_ssim(ndct, pred, mask)

        if ps is not None:
            psnr_list.append(ps)
            ssim_list.append(ss)

# =============================
# RESULTS
# =============================
print("\n===== RED-CNN RESULTS =====")

if len(psnr_list) == 0:
    print("❌ No valid matches found")
else:
    print(f"PSNR: {np.mean(psnr_list):.2f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print(f"Used: {len(psnr_list)}")
    print(f"Missing: {missing}")