import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity


# ==========================
# PATHS
# ==========================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
ENH_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"
CNN_ROOT = r"D:\CT_Datasets\Phase3_CNN_Refined"


# ==========================
# PATIENT SPLIT
# ==========================

VALIDATION_PATIENTS = [
"LIDC-IDRI-0021",
"LIDC-IDRI-0022",
"LIDC-IDRI-0023"
]

TEST_PATIENTS = [
"LIDC-IDRI-0024",
"LIDC-IDRI-0025",
"LIDC-IDRI-0026"
]


# ==========================
# UTILITIES
# ==========================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


def compute_psnr(gt, pred, mask):

    gt = gt[mask == 1]
    pred = pred[mask == 1]

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


# ==========================
# STORAGE
# ==========================

val_psnr = []
val_ssim = []

test_psnr = []
test_ssim = []

val_slices = 0
test_slices = 0


# ==========================
# LOOP
# ==========================

for root, dirs, files in os.walk(CNN_ROOT):

    relative = os.path.relpath(root, CNN_ROOT)

    patient_id = None

    for part in relative.split(os.sep):
        if part.startswith("LIDC-IDRI-"):
            patient_id = part
            break


    for file in files:

        if not file.endswith("_cnn_refined.png"):
            continue


        base = file.replace("_cnn_refined.png", "")

        cnn_path = os.path.join(root, file)

        ndct_path = os.path.join(
            LDCT_ROOT,
            relative,
            base + "_ndct.png"
        )

        mask_path = os.path.join(
            ENH_ROOT,
            relative,
            base + "_lung_mask.png"
        )

        if not all(os.path.exists(p) for p in [ndct_path, mask_path]):
            continue


        cnn = load_image(cnn_path)
        ndct = load_image(ndct_path)
        mask = (load_image(mask_path) > 0.5).astype(np.uint8)


        # ensure same size
        if cnn.shape != ndct.shape:
            cnn = cv2.resize(cnn, (ndct.shape[1], ndct.shape[0]))

        if mask.shape != ndct.shape:
            mask = cv2.resize(mask, (ndct.shape[1], ndct.shape[0]))


        psnr = compute_psnr(ndct, cnn, mask)
        ssim = compute_ssim(ndct, cnn, mask)


        # ==========================
        # VALIDATION
        # ==========================

        if patient_id in VALIDATION_PATIENTS:

            val_psnr.append(psnr)
            val_ssim.append(ssim)
            val_slices += 1


        # ==========================
        # TEST
        # ==========================

        if patient_id in TEST_PATIENTS:

            test_psnr.append(psnr)
            test_ssim.append(ssim)
            test_slices += 1


# ==========================
# FINAL RESULTS
# ==========================

print("\n==============================")
print("CNN REFINED RESULTS")
print("==============================\n")


print("Validation Patients:", VALIDATION_PATIENTS)
print("Validation slices:", val_slices)

print(
    "Validation PSNR:",
    np.mean(val_psnr),
    "±",
    np.std(val_psnr)
)

print(
    "Validation SSIM:",
    np.mean(val_ssim),
    "±",
    np.std(val_ssim)
)


print("\n--------------------------------\n")


print("Test Patients:", TEST_PATIENTS)
print("Test slices:", test_slices)

print(
    "Test PSNR:",
    np.mean(test_psnr),
    "±",
    np.std(test_psnr)
)

print(
    "Test SSIM:",
    np.mean(test_ssim),
    "±",
    np.std(test_ssim)
)