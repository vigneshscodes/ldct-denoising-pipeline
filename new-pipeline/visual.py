import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG (EDIT ONLY THIS)
# =========================================================
PATIENT_ID = "LIDC-IDRI-0024"
SLICE_NUM  = "1-053"

LDCT_ROOT   = r"D:\CT_Datasets\LDCT"
REGION_ROOT = r"D:\CT_Datasets\Phase2_Output"
CNN_ROOT    = r"D:\CT_Datasets\Phase3_CNN_Refined"
SEG_ROOT    = r"D:\CT_Datasets\Segmentation"

# =========================================================
# FIND FILE HELPER
# =========================================================
def find_file(root, patient_id, slice_num, suffix):
    for r, _, files in os.walk(root):
        if patient_id in r:
            for f in files:
                if f.endswith(suffix) and f.startswith(slice_num):
                    return os.path.join(r, f)
    return None

# =========================================================
# LOAD IMAGES
# =========================================================
ldct_path   = find_file(LDCT_ROOT, PATIENT_ID, SLICE_NUM, "_ldct.png")
ndct_path   = find_file(LDCT_ROOT, PATIENT_ID, SLICE_NUM, "_ndct.png")
region_path = find_file(REGION_ROOT, PATIENT_ID, SLICE_NUM, "_region.png")
cnn_path    = find_file(CNN_ROOT, PATIENT_ID, SLICE_NUM, "_cnn_refined.png")

if None in [ldct_path, ndct_path, region_path, cnn_path]:
    raise ValueError("Some images not found. Check patient/slice.")

imgs = {
    "LDCT": cv2.imread(ldct_path, 0),
    "Region": cv2.imread(region_path, 0),
    "CNN": cv2.imread(cnn_path, 0),
    "NDCT": cv2.imread(ndct_path, 0)
}

# =========================================================
# FULL COMPARISON
# =========================================================
plt.figure(figsize=(12,4))
for i, (title, img) in enumerate(imgs.items()):
    plt.subplot(1,4,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.savefig("figure_full_comparison.png", dpi=300)
plt.show()

# =========================================================
# ZOOMED PATCH (EDIT IF NEEDED)
# =========================================================
x, y, w, h = 150, 150, 120, 120

plt.figure(figsize=(12,4))
for i, (title, img) in enumerate(imgs.items()):
    crop = img[y:y+h, x:x+w]
    plt.subplot(1,4,i+1)
    plt.imshow(crop, cmap='gray')
    plt.title(title + " (Zoom)")
    plt.axis('off')

plt.tight_layout()
plt.savefig("figure_zoomed_patch.png", dpi=300)
plt.show()

# =========================================================
# MASKS
# =========================================================
lung_path = find_file(SEG_ROOT, PATIENT_ID, SLICE_NUM, "_lung_mask.png")
bone_path = find_file(SEG_ROOT, PATIENT_ID, SLICE_NUM, "_bone_mask.png")
soft_path = find_file(SEG_ROOT, PATIENT_ID, SLICE_NUM, "_soft_mask.png")

if None in [lung_path, bone_path, soft_path]:
    raise ValueError(" Mask images not found.")

masks = {
    "Lung Mask": cv2.imread(lung_path, 0),
    "Bone Mask": cv2.imread(bone_path, 0),
    "Soft Tissue Mask": cv2.imread(soft_path, 0)
}

plt.figure(figsize=(10,3))
for i, (title, img) in enumerate(masks.items()):
    plt.subplot(1,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.savefig("figure_masks.png", dpi=300)
plt.show()

print("\n All figures generated successfully!")

