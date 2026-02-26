# enhancement.py

import numpy as np
import pywt
import cv2


# ---------------------------------
# Noise Estimation (Robust)
# ---------------------------------
def estimate_noise(img_norm):
    """
    Estimate noise sigma using median absolute deviation (MAD)
    on high-frequency wavelet coefficients.
    """
    coeffs = pywt.wavedec2(img_norm, 'db1', level=1)
    cH, cV, cD = coeffs[1]

    # Use diagonal detail coefficients
    sigma = np.median(np.abs(cD)) / 0.6745
    return sigma


# ---------------------------------
# Adaptive Wavelet Denoising
# ---------------------------------
def wavelet_denoise(img_norm, wavelet='db1', level=2):

    sigma = estimate_noise(img_norm)

    # Universal threshold
    threshold = sigma * np.sqrt(2 * np.log(img_norm.size))

    coeffs = pywt.wavedec2(img_norm, wavelet, level=level)

    coeffs_thresh = [coeffs[0]]

    for detail in coeffs[1:]:
        cH, cV, cD = detail
        coeffs_thresh.append((
            pywt.threshold(cH, threshold, mode='soft'),
            pywt.threshold(cV, threshold, mode='soft'),
            pywt.threshold(cD, threshold, mode='soft')
        ))

    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    denoised = np.clip(denoised, 0, 1)

    return denoised


# ---------------------------------
# Controlled CLAHE
# ---------------------------------
def apply_clahe(img_norm,
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                blend_factor=0.4):
    """
    CLAHE with blending to avoid over-amplification.
    blend_factor: 0 = no CLAHE, 1 = full CLAHE
    """

    img_uint8 = (img_norm * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    clahe_img = clahe.apply(img_uint8)
    clahe_img = clahe_img.astype(np.float32) / 255.0

    # Blend with original to prevent noise amplification
    enhanced = (1 - blend_factor) * img_norm + blend_factor * clahe_img
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


# ---------------------------------
# Combined Enhancement
# ---------------------------------
def enhance_ldct(img_norm):

    # Step 1: Adaptive wavelet denoising
    wavelet_img = wavelet_denoise(img_norm)

    # Step 2: Mild contrast enhancement with blending
    enhanced_img = apply_clahe(wavelet_img,
                               clip_limit=2.0,
                               blend_factor=0.4)

    return enhanced_img