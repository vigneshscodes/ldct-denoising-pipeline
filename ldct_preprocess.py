import os
import numpy as np
import pydicom
import pywt
import cv2

input_root = r"D:\CT_Datasets\LDCT"
output_root = r"D:\CT_Datasets\LDCT_Processed"

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".dcm"):

            in_path = os.path.join(root, file)

            rel = os.path.relpath(root, input_root)
            out_folder = os.path.join(output_root, rel)
            os.makedirs(out_folder, exist_ok=True)

            out_path = os.path.join(out_folder, file)

            ds = pydicom.dcmread(in_path)
            img = ds.pixel_array.astype(np.float32)

            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            img_hu = img * slope + intercept

            # normalize safely
            img_min = img_hu.min()
            img_max = img_hu.max()
            img_norm = (img_hu - img_min) / (img_max - img_min + 1e-8)

            # wavelet denoise
            coeffs = pywt.wavedec2(img_norm, 'db1', level=2)
            coeffs_thresh = [coeffs[0]]

            for detail in coeffs[1:]:
                cH, cV, cD = detail
                coeffs_thresh.append((
                    pywt.threshold(cH, 0.02),
                    pywt.threshold(cV, 0.02),
                    pywt.threshold(cD, 0.02)
                ))

            denoised = pywt.waverec2(coeffs_thresh, 'db1')
            denoised = np.clip(denoised, 0, 1)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=1.0)
            enhanced = clahe.apply((denoised * 255).astype(np.uint8))

            # convert back to HU
            enhanced = enhanced.astype(np.float32) / 255.0
            enhanced = enhanced * (img_max - img_min) + img_min
            enhanced = (enhanced - intercept) / slope

            ds.PixelData = enhanced.astype(ds.pixel_array.dtype).tobytes()
            ds.save_as(out_path)

print("Wavelet + CLAHE processing completed.")
