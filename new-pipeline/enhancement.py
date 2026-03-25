import numpy as np
import cv2


# ---------------------------------
# CLAHE in 16-bit (avoid banding)
# ---------------------------------
def apply_clahe_float(img_norm, clip_limit=1.2):

    img_uint16 = (img_norm * 65535).astype(np.uint16)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(img_uint16)
    clahe_img = clahe_img.astype(np.float32) / 65535.0

    return clahe_img


# ---------------------------------
# Mild Sharpening (safe)
# ---------------------------------
def mild_sharpen(img, alpha=0.25):

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharpened = img + alpha * (img - blurred)

    return np.clip(sharpened, 0, 1)


# ---------------------------------
# FINAL ENHANCEMENT
# ---------------------------------
def enhance_ldct(img_norm):

    clahe_img = apply_clahe_float(img_norm, clip_limit=1.2)

    enhanced = mild_sharpen(clahe_img, alpha=0.25)

    return enhanced