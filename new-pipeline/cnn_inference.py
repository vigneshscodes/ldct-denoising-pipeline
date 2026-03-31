import os
import cv2
import torch
import numpy as np
import torch.nn as nn


# ==============================
# PATHS (FIXED)
# ==============================

MODEL_PATH = r"D:\CT_Datasets\best_model.pth"

LDCT_ROOT   = r"D:\CT_Datasets\LDCT"
REGION_ROOT = r"D:\CT_Datasets\Phase2_Output"   # ✅ FIXED
SEG_ROOT    = r"D:\CT_Datasets\Segmentation"    # ✅ FIXED

OUTPUT_ROOT = r"D:\CT_Datasets\Phase3_CNN_Refined"


# ==============================
# TEST PATIENTS ONLY (FINAL EVAL)
# ==============================

EVAL_PATIENTS = [
    "LIDC-IDRI-0024",
    "LIDC-IDRI-0025",
    "LIDC-IDRI-0026"
]


# ==============================
# MODEL
# ==============================

class RefinementCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# LOAD MODEL
# ==============================

device = torch.device("cpu")

model = RefinementCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ==============================
# IMAGE UTILS
# ==============================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)


# ==============================
# INFERENCE LOOP (FINAL)
# ==============================

for root, dirs, files in os.walk(REGION_ROOT):

    relative = os.path.relpath(root, REGION_ROOT)

    # detect patient
    patient_id = None
    for part in relative.split(os.sep):
        if part.startswith("LIDC-IDRI-"):
            patient_id = part
            break

    if patient_id not in EVAL_PATIENTS:
        continue

    for file in files:

        # ✅ FIXED filename
        if not file.endswith("_region.png"):
            continue

        base = file.replace("_region.png", "")

        region_path = os.path.join(root, file)

        ldct_path = os.path.join(LDCT_ROOT, relative, base + "_ldct.png")
        mask_path = os.path.join(SEG_ROOT, relative, base + "_lung_mask.png")

        # check paths
        if not all(os.path.exists(p) for p in [ldct_path, mask_path]):
            continue

        ldct   = load_image(ldct_path)
        region = load_image(region_path)
        mask   = load_image(mask_path)

        if ldct is None or region is None or mask is None:
            continue

        mask = (mask > 0.5).astype(np.float32)

        # skip small lung slices (IMPORTANT)
        if np.sum(mask) / mask.size < 0.05:
            continue

        # input tensor
        x = np.stack([ldct, region, mask], axis=0)
        x = torch.tensor(x).unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            pred = model(x)

        pred = pred.squeeze().cpu().numpy()

        # save
        save_path = os.path.join(
            OUTPUT_ROOT,
            relative,
            base + "_cnn_refined.png"
        )

        save_image(pred, save_path)

        print("Saved:", save_path)


print("\n✅ CNN inference completed for TEST patients")