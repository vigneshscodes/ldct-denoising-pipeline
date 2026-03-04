import os
import cv2
import torch
import numpy as np
import torch.nn as nn


# ==============================
# PATHS
# ==============================

MODEL_PATH = r"D:\CT_Datasets\cnn_refinement_model.pth"

LDCT_ROOT = r"D:\CT_Datasets\LDCT"
REGION_ROOT = r"D:\CT_Datasets\Phase2_RegionAdaptive"
MASK_ROOT = r"D:\CT_Datasets\LDCT_Enhanced"

OUTPUT_ROOT = r"D:\CT_Datasets\Phase3_CNN_Refined"


# ==============================
# MODEL DEFINITION
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
    return img.astype(np.float32) / 255.0


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)


# ==============================
# INFERENCE LOOP
# ==============================

for root, dirs, files in os.walk(REGION_ROOT):

    relative = os.path.relpath(root, REGION_ROOT)

    for file in files:

        if not file.endswith("_region_adaptive.png"):
            continue

        base = file.replace("_region_adaptive.png", "")

        region_path = os.path.join(root, file)

        ldct_path = os.path.join(
            LDCT_ROOT,
            relative,
            base + "_ldct.png"
        )

        mask_path = os.path.join(
            MASK_ROOT,
            relative,
            base + "_lung_mask.png"
        )

        if not all(os.path.exists(p) for p in [ldct_path, mask_path]):
            continue


        ldct = load_image(ldct_path)
        region = load_image(region_path)
        mask = load_image(mask_path)

        mask = (mask > 0.5).astype(np.float32)


        x = np.stack([ldct, region, mask], axis=0)

        x = torch.tensor(x).unsqueeze(0).to(device)


        with torch.no_grad():
            pred = model(x)

        pred = pred.squeeze().cpu().numpy()


        save_path = os.path.join(
            OUTPUT_ROOT,
            relative,
            base + "_cnn_refined.png"
        )

        save_image(pred, save_path)

        print("Saved:", save_path)

print("CNN inference completed.")