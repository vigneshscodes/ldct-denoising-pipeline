import os
import cv2
import numpy as np

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
REGION_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"
MASK_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"

OUTPUT_ROOT = r"D:\CT_Datasets\CNN_Dataset_PatientWise"

TRAIN_INPUT = os.path.join(OUTPUT_ROOT, "train_inputs")
TRAIN_TARGET = os.path.join(OUTPUT_ROOT, "train_targets")
TEST_INPUT = os.path.join(OUTPUT_ROOT, "test_inputs")
TEST_TARGET = os.path.join(OUTPUT_ROOT, "test_targets")

os.makedirs(TRAIN_INPUT, exist_ok=True)
os.makedirs(TRAIN_TARGET, exist_ok=True)
os.makedirs(TEST_INPUT, exist_ok=True)
os.makedirs(TEST_TARGET, exist_ok=True)


TRAIN_PATIENTS = [
"LIDC-IDRI-0001","LIDC-IDRI-0002","LIDC-IDRI-0003","LIDC-IDRI-0004","LIDC-IDRI-0005",
"LIDC-IDRI-0006","LIDC-IDRI-0007","LIDC-IDRI-0008","LIDC-IDRI-0009","LIDC-IDRI-0010",
"LIDC-IDRI-0011","LIDC-IDRI-0012","LIDC-IDRI-0013","LIDC-IDRI-0014","LIDC-IDRI-0015"
]

TEST_PATIENTS = [
"LIDC-IDRI-0016","LIDC-IDRI-0017","LIDC-IDRI-0018","LIDC-IDRI-0019","LIDC-IDRI-0020"
]


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


sample_index = 0

for root, dirs, files in os.walk(LDCT_ROOT):

    relative = os.path.relpath(root, LDCT_ROOT)

    patient_id = None

    for part in relative.split(os.sep):
        if part.startswith("LIDC-IDRI"):
            patient_id = part
            break

    if patient_id is None:
        continue

    for file in files:

        if not file.endswith("_ldct.png"):
            continue

        base = file.replace("_ldct.png", "")

        ldct_path = os.path.join(root, base + "_ldct.png")
        ndct_path = os.path.join(root, base + "_ndct.png")

        region_path = os.path.join(
            REGION_ROOT,
            relative,
            base + "_region_adaptive.png"
        )

        mask_path = os.path.join(
            MASK_ROOT,
            relative,
            base + "_lung_mask.png"
        )

        if not all(os.path.exists(p) for p in
                   [ldct_path, ndct_path, region_path, mask_path]):
            continue

        ldct = load_image(ldct_path)
        region = load_image(region_path)
        mask = load_image(mask_path)
        ndct = load_image(ndct_path)

        mask = (mask > 0.5).astype(np.float32)

        if np.sum(mask) / mask.size < 0.05:
            continue

        input_tensor = np.stack([ldct, region, mask], axis=0)
        target_tensor = ndct[np.newaxis, :, :]

        sample_name = f"sample_{sample_index:06d}.npy"

        if patient_id in TRAIN_PATIENTS:

            np.save(os.path.join(TRAIN_INPUT, sample_name), input_tensor)
            np.save(os.path.join(TRAIN_TARGET, sample_name), target_tensor)

        elif patient_id in TEST_PATIENTS:

            np.save(os.path.join(TEST_INPUT, sample_name), input_tensor)
            np.save(os.path.join(TEST_TARGET, sample_name), target_tensor)

        sample_index += 1


print("Dataset creation finished.")
print("Total samples:", sample_index)