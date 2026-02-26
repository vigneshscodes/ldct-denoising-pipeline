# preprocessing.py

import pydicom
import numpy as np


# ------------------------------
# Load DICOM and convert to HU
# ------------------------------
def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    slope = getattr(ds, "RescaleSlope", 1)
    intercept = getattr(ds, "RescaleIntercept", 0)

    img_hu = img * slope + intercept
    return ds, img_hu


# ------------------------------
# Apply fixed lung window
# ------------------------------
def apply_lung_window(img_hu, window_level=-600, window_width=1500):

    lower = window_level - (window_width / 2)
    upper = window_level + (window_width / 2)

    img_windowed = np.clip(img_hu, lower, upper)

    # Normalize to [0,1]
    img_norm = (img_windowed - lower) / (upper - lower)

    return img_norm, lower, upper