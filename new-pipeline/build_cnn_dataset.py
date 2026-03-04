import os
import cv2
import numpy as np

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
REGION_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"
MASK_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"

OUTPUT_ROOT = r"D:\CT_Datasets\CNN_Dataset"

INPUT_DIR = os.path.join(OUTPUT_ROOT, "inputs")
TARGET_DIR = os.path.join(OUTPUT_ROOT, "targets")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

sample_index = 0


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


for root, dirs, files in os.walk(LDCT_ROOT):

    for file in files:

        if not file.endswith("_ldct.png"):
            continue

        base = file.replace("_ldct.png", "")

        relative = os.path.relpath(root, LDCT_ROOT)

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

        # Skip slices with very small lung
        if np.sum(mask) / mask.size < 0.05:
            continue

        input_tensor = np.stack([ldct, region, mask], axis=0)
        target_tensor = ndct[np.newaxis, :, :]

        sample_name = f"sample_{sample_index:06d}.npy"

        np.save(os.path.join(INPUT_DIR, sample_name), input_tensor)
        np.save(os.path.join(TARGET_DIR, sample_name), target_tensor)

        sample_index += 1

print("Total CNN samples created:", sample_index)