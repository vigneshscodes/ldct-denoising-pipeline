import numpy as np
import pywt
import cv2


# ---------------------------------
# Noise Estimation (Robust)
# ---------------------------------
def estimate_noise(img_norm):
    coeffs = pywt.wavedec2(img_norm, 'db1', level=1)
    _, _, cD = coeffs[1]
    sigma = np.median(np.abs(cD)) / 0.6745
    return sigma


# ---------------------------------
# Adaptive Wavelet Denoising
# ---------------------------------
def wavelet_denoise(img_norm, wavelet='db1', level=1):

    sigma = estimate_noise(img_norm)

    threshold = 0.8 * sigma * np.sqrt(2 * np.log(img_norm.size))

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
    return np.clip(denoised, 0, 1)


# ---------------------------------
# Controlled CLAHE
# ---------------------------------
def apply_clahe(img_norm,
                clip_limit=1.5,
                tile_grid_size=(16, 16),
                blend_factor=0.25):

    img_uint8 = (img_norm * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    clahe_img = clahe.apply(img_uint8)
    clahe_img = clahe_img.astype(np.float32) / 255.0

    enhanced = (1 - blend_factor) * img_norm + blend_factor * clahe_img
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


# ---------------------------------
# Combined Enhancement (FIXED)
# ---------------------------------
def enhance_ldct(img_norm):

    # Slightly stronger denoising
    wavelet_img = wavelet_denoise(img_norm, level=1)

    enhanced_img = apply_clahe(
        wavelet_img,
        clip_limit=1.6,
        tile_grid_size=(16, 16),
        blend_factor=0.22
    )

    return enhanced_img