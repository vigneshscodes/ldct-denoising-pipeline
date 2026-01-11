import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# ----------------------------
# Load image (proxy for now)
# ----------------------------
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

# ----------------------------
# Normalization
# ----------------------------
def normalize_image(img):
    img = img.astype("float32")
    img_norm = img / 255.0
    return img_norm

# ----------------------------
# Wavelet denoising
# ----------------------------
def wavelet_denoise(img_norm, threshold=0.08):
    coeffs = pywt.wavedec2(img_norm, 'db1', level=2)
    coeffs = list(coeffs)

    for i in range(1, len(coeffs)):
        coeffs[i] = tuple(
            pywt.threshold(c, threshold, mode='soft') for c in coeffs[i]
        )

    img_denoised = pywt.waverec2(coeffs, 'db1')
    return img_denoised

# ----------------------------
# CLAHE enhancement
# ----------------------------
def apply_clahe(img):
    img_uint8 = (img * 255).astype("uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)
    return img_clahe

def run_preprocessing_pipeline(image_path):
    img = load_image(image_path)
    img_norm = normalize_image(img)
    img_wavelet = wavelet_denoise(img_norm)
    img_final = apply_clahe(img_wavelet)
    return img, img_norm, img_wavelet, img_final

# ----------------------------
# DICOM loader placeholder
# ----------------------------
def load_dicom_volume(patient_folder):
    """
    Placeholder for future integration.
    Will:
    - Read all .dcm files in patient folder
    - Sort by InstanceNumber
    - Convert to HU using RescaleSlope & Intercept
    - Return 3D volume
    """
    pass

# ----------------------------
# Segmentation stub (pre-trained UNet)
# ----------------------------
def segment_lung(slice_image):
    """
    Uses pre-trained UNet to generate lung mask.
    Model: model_100_epoch.pth (from Kaggle)
    Input: 2D CT slice
    Output: binary lung mask
    """
    pass


if __name__ == "__main__":
    image_path = "data/ct_images/ID00007637202177411956430_0.jpg"

    img, img_norm, img_wavelet, img_final = run_preprocessing_pipeline(image_path)

    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.show()

    plt.imshow(img_norm, cmap="gray")
    plt.title("Normalized")
    plt.axis("off")
    plt.show()

    plt.imshow(img_wavelet, cmap="gray")
    plt.title("Wavelet Denoised")
    plt.axis("off")
    plt.show()

    plt.imshow(img_final, cmap="gray")
    plt.title("Wavelet + CLAHE")
    plt.axis("off")
    plt.show()
