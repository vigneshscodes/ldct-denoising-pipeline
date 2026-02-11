import os
import numpy as np
import pydicom

input_root = r"D:\CT_Datasets\NDCT"
output_root = r"D:\CT_Datasets\LDCT"
dose_factor = 0.25   # 25% dose simulation
gaussian_sigma = 0.01
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".dcm"):
            input_path = os.path.join(root, file)
            # Create corresponding output folder
            relative_path = os.path.relpath(root, input_root)
            output_folder = os.path.join(output_root, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, file)
            # Read DICOM
            ds = pydicom.dcmread(input_path)
            img = ds.pixel_array.astype(np.float32)
            # Handle HU conversion
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            img = img * slope + intercept
            # Normalize
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Dose scaling
            scaled = img_norm * dose_factor
            # Poisson noise (photon noise)
            poisson = np.random.poisson(scaled * 1000) / 1000.0
            # Gaussian detector noise
            gaussian = np.random.normal(0, gaussian_sigma, img.shape)
            ldct = poisson + gaussian
            ldct = np.clip(ldct, 0, 1)
            # Convert back to original HU range
            ldct = ldct * (img.max() - img.min()) + img.min()
            # Reverse slope/intercept for saving
            ldct = (ldct - intercept) / slope
            ldct = ldct.astype(ds.pixel_array.dtype)
            ds.PixelData = ldct.tobytes()
            ds.save_as(output_path)
print("LDCT generation completed.")
