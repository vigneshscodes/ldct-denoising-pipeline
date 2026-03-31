import os
import cv2
import numpy as np

# ==========================
# PATHS (MATCH EVALUATION)
# ==========================

LDCT_ROOT   = r"D:\CT_Datasets\LDCT"
SEG_ROOT    = r"D:\CT_Datasets\Segmentation"
PHASE2_ROOT = r"D:\CT_Datasets\Phase2_Output"

OUTPUT_ROOT = r"D:\CT_Datasets\CNN_Dataset_PatientWise"

# ==========================
# OUTPUT FOLDERS
# ==========================

TRAIN_INPUT = os.path.join(OUTPUT_ROOT, "train_inputs")
TRAIN_TARGET = os.path.join(OUTPUT_ROOT, "train_targets")

VAL_INPUT = os.path.join(OUTPUT_ROOT, "val_inputs")
VAL_TARGET = os.path.join(OUTPUT_ROOT, "val_targets")

TEST_INPUT = os.path.join(OUTPUT_ROOT, "test_inputs")
TEST_TARGET = os.path.join(OUTPUT_ROOT, "test_targets")

for p in [TRAIN_INPUT, TRAIN_TARGET, VAL_INPUT, VAL_TARGET, TEST_INPUT, TEST_TARGET]:
    os.makedirs(p, exist_ok=True)

# ==========================
# PATIENT SPLIT
# ==========================

TRAIN_PATIENTS = [f"LIDC-IDRI-{i:04d}" for i in range(1, 21)]
VAL_PATIENTS   = [f"LIDC-IDRI-{i:04d}" for i in range(21, 24)]
TEST_PATIENTS  = [f"LIDC-IDRI-{i:04d}" for i in range(24, 27)]

# ==========================
# IMAGE LOADER
# ==========================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0

# ==========================
# COUNTERS
# ==========================

train_count = 0
val_count = 0
test_count = 0

# ==========================
# MAIN LOOP (MATCHED LOGIC)
# ==========================

for root, dirs, files in os.walk(LDCT_ROOT):

    # match patient
    if not any(p in root for p in TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS):
        continue

    relative_path = os.path.relpath(root, LDCT_ROOT)

    seg_root = os.path.join(SEG_ROOT, relative_path)
    den_root = os.path.join(PHASE2_ROOT, relative_path)

    for file in files:

        if not file.endswith("_ldct.png"):
            continue

        base = file.replace("_ldct.png", "")

        ldct_path = os.path.join(root, base + "_ldct.png")
        ndct_path = os.path.join(root, base + "_ndct.png")
        mask_path = os.path.join(seg_root, base + "_lung_mask.png")
        region_path = os.path.join(den_root, base + "_region.png")

        # check all exist
        if not all(os.path.exists(p) for p in [ldct_path, ndct_path, mask_path, region_path]):
            continue

        ldct = load_image(ldct_path)
        ndct = load_image(ndct_path)
        region = load_image(region_path)
        mask = load_image(mask_path)

        if ldct is None or ndct is None or region is None or mask is None:
            continue

        mask = (mask > 0.5).astype(np.float32)

        # same filtering as evaluation
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        if np.sum(mask) / mask.size < 0.05:
            continue

        # ==========================
        # INPUT / TARGET
        # ==========================

        input_tensor = np.stack([ldct, region, mask], axis=0)
        target_tensor = ndct[np.newaxis, :, :]

        # detect patient
        patient_id = None
        for part in root.split(os.sep):
            if part.startswith("LIDC-IDRI-"):
                patient_id = part
                break

        # ==========================
        # SAVE
        # ==========================

        if patient_id in TRAIN_PATIENTS:
            name = f"train_{train_count:06d}.npy"
            np.save(os.path.join(TRAIN_INPUT, name), input_tensor)
            np.save(os.path.join(TRAIN_TARGET, name), target_tensor)
            train_count += 1

        elif patient_id in VAL_PATIENTS:
            name = f"val_{val_count:06d}.npy"
            np.save(os.path.join(VAL_INPUT, name), input_tensor)
            np.save(os.path.join(VAL_TARGET, name), target_tensor)
            val_count += 1

        elif patient_id in TEST_PATIENTS:
            name = f"test_{test_count:06d}.npy"
            np.save(os.path.join(TEST_INPUT, name), input_tensor)
            np.save(os.path.join(TEST_TARGET, name), target_tensor)
            test_count += 1

# ==========================
# SUMMARY
# ==========================

print("\n✅ Dataset creation finished.")
print("Train samples:", train_count)
print("Val samples:", val_count)
print("Test samples:", test_count)