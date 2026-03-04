import os
import cv2
import numpy as np

# ==========================
# PATHS
# ==========================

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
REGION_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"
MASK_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"

NDCT_ROOT = r"D:\CT_Datasets\LDCT"

OUTPUT_ROOT = r"D:\CT_Datasets\CNN_Dataset_PatientWise"

train_input_dir = os.path.join(OUTPUT_ROOT, "train_inputs")
train_target_dir = os.path.join(OUTPUT_ROOT, "train_targets")

test_input_dir = os.path.join(OUTPUT_ROOT, "test_inputs")
test_target_dir = os.path.join(OUTPUT_ROOT, "test_targets")

os.makedirs(train_input_dir, exist_ok=True)
os.makedirs(train_target_dir, exist_ok=True)
os.makedirs(test_input_dir, exist_ok=True)
os.makedirs(test_target_dir, exist_ok=True)


# ==========================
# TRAIN / TEST PATIENT SPLIT
# ==========================

TRAIN_PATIENTS = [
"LIDC-IDRI-0001","LIDC-IDRI-0002","LIDC-IDRI-0003","LIDC-IDRI-0004","LIDC-IDRI-0005",
"LIDC-IDRI-0006","LIDC-IDRI-0007","LIDC-IDRI-0008","LIDC-IDRI-0009","LIDC-IDRI-0010",
"LIDC-IDRI-0011","LIDC-IDRI-0012","LIDC-IDRI-0013","LIDC-IDRI-0014","LIDC-IDRI-0015"
]

TEST_PATIENTS = [
"LIDC-IDRI-0016","LIDC-IDRI-0017","LIDC-IDRI-0018","LIDC-IDRI-0019","LIDC-IDRI-0020"
]


# ==========================
# UTILITIES
# ==========================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


sample_id = 0

# ==========================
# DATASET CREATION LOOP
# ==========================

for root, dirs, files in os.walk(REGION_ROOT):

    relative = os.path.relpath(root, REGION_ROOT)

    # detect patient id
    patient_id = None

    for part in relative.split(os.sep):
        if part.startswith("LIDC-IDRI"):
            patient_id = part
            break

    if patient_id is None:
        continue

    for file in files:

        if not file.endswith("_region_adaptive.png"):
            continue

        base = file.replace("_region_adaptive.png", "")

        region_path = os.path.join(root, file)

        ldct_path = os.path.join(LDCT_ROOT, relative, base + "_ldct.png")
        ndct_path = os.path.join(NDCT_ROOT, relative, base + "_ndct.png")
        mask_path = os.path.join(MASK_ROOT, relative, base + "_lung_mask.png")

        if not all(os.path.exists(p) for p in [ldct_path, ndct_path, mask_path]):
            continue

        ldct = load_image(ldct_path)
        region = load_image(region_path)
        mask = load_image(mask_path)

        mask = (mask > 0.5).astype(np.float32)

        x = np.stack([ldct, region, mask], axis=0)
        y = np.expand_dims(load_image(ndct_path), axis=0)

        name = f"sample_{sample_id:06d}.npy"

        if patient_id in TRAIN_PATIENTS:

            np.save(os.path.join(train_input_dir, name), x)
            np.save(os.path.join(train_target_dir, name), y)

        elif patient_id in TEST_PATIENTS:

            np.save(os.path.join(test_input_dir, name), x)
            np.save(os.path.join(test_target_dir, name), y)

        sample_id += 1


print("Dataset creation finished.")
print("Total samples:", sample_id)