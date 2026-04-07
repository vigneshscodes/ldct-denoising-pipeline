import os
import shutil
import cv2
import numpy as np

SRC_LDCT = r"D:\CT_Datasets\LDCT"
SRC_SEG  = r"D:\CT_Datasets\Segmentation"
DEST     = r"D:\CT_CleanRed"

splits = ["train", "val", "test"]

for split in splits:
    src_split = os.path.join(SRC_LDCT, split)
    dest_split = os.path.join(DEST, split)

    os.makedirs(dest_split, exist_ok=True)

    for root, _, files in os.walk(src_split):

        # extract patient ID
        patient_id = None
        for part in root.split(os.sep):
            if part.startswith("LIDC-IDRI-"):
                patient_id = part
                break

        if patient_id is None:
            continue

        relative = os.path.relpath(root, SRC_LDCT)
        seg_root = os.path.join(SRC_SEG, relative)

        patient_folder = os.path.join(dest_split, patient_id)
        os.makedirs(patient_folder, exist_ok=True)

        for f in files:
            if not f.endswith("_ldct.png"):
                continue

            base = f.replace("_ldct.png", "")

            ldct = os.path.join(root, base + "_ldct.png")
            ndct = os.path.join(root, base + "_ndct.png")
            mask = os.path.join(seg_root, base + "_lung_mask.png")

            if not (os.path.exists(ndct) and os.path.exists(mask)):
                continue

            # 🔥 LOAD MASK
            m = cv2.imread(mask, 0)
            if m is None:
                continue

            m = (m > 127).astype(np.uint8)

            # 🔥 APPLY 5% RULE
            if np.sum(m) / m.size < 0.05:
                continue

            # copy
            shutil.copy(ldct, os.path.join(patient_folder, base + "_ldct.png"))
            shutil.copy(ndct, os.path.join(patient_folder, base + "_ndct.png"))

print("✅ FILTERED DATASET CREATED (5% RULE APPLIED)")
