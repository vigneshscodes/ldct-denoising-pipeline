# enhancement.py

import numpy as np
import pywt
import cv2


# ------------------------------
# Wavelet shrinkage
# ------------------------------
def wavelet_denoise(img_norm, wavelet='db1', level=2, threshold=0.02):

    coeffs = pywt.wavedec2(img_norm, wavelet, level=level)

    coeffs_thresh = [coeffs[0]]

    for detail in coeffs[1:]:
        cH, cV, cD = detail
        coeffs_thresh.append((
            pywt.threshold(cH, threshold),
            pywt.threshold(cV, threshold),
            pywt.threshold(cD, threshold)
        ))

    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    denoised = np.clip(denoised, 0, 1)

    return denoised


# ------------------------------
# CLAHE enhancement
# ------------------------------
def apply_clahe(img_norm, clip_limit=2.0):

    img_uint8 = (img_norm * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(img_uint8)
    enhanced = enhanced.astype(np.float32) / 255.0

    return enhanced


# ------------------------------
# Combined enhancement
# ------------------------------
def enhance_ldct(img_norm):

    wavelet_img = wavelet_denoise(img_norm)
    enhanced_img = apply_clahe(wavelet_img)

    return enhanced_img