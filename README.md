# Low-Dose CT Denoising using Classical Methods and Lightweight CNN Refinement

## Overview

This project implements a modular pipeline for **Low-Dose CT (LDCT) denoising** using a combination of classical image processing and a lightweight convolutional neural network (CNN). The goal is to reduce noise in simulated LDCT scans while preserving anatomical structures in lung regions.

The pipeline integrates physics-based noise simulation, segmentation-guided denoising, and CNN refinement to produce high-quality reconstructed CT images.

---

## Dataset

CT scans are obtained from the **LIDC-IDRI dataset** available on The Cancer Imaging Archive (TCIA).

Dataset usage:

* **Training:** 20 patients
* **Validation:** 3 patients
* **Testing:** 3 patients

All evaluations are performed **patient-wise** to avoid data leakage.

---

## Pipeline

The processing pipeline consists of the following stages:

1. **NDCT Input**
2. **LDCT Simulation**

   * Poisson noise model for low-dose simulation
3. **Image Enhancement**

   * Wavelet denoising
   * CLAHE contrast enhancement
4. **Segmentation**

   * UNet-based lung segmentation
5. **Phase 1: Classical Denoising**

   * Bilateral filter
   * Non-local means
   * BM3D
6. **Phase 2: Region-Adaptive Denoising**

   * Lung → Bilateral
   * Bone → NLM
   * Soft Tissue → BM3D
7. **Phase 3: CNN Refinement**

   * Lightweight 3-layer CNN
   * Inputs: LDCT + Region-Adaptive Output + Lung Mask
   * Loss: L1 + SSIM
8. **Evaluation**

   * PSNR
   * SSIM

---

## Project Structure

```
CT_Datasets
├── NDCT
├── NDCT_Eval
├── LDCT
├── LDCT_Enhanced
├── Phase1_Classical
├── Phase2_RegionAdaptive
├── Phase3_CNN_Refined
└── CNN_Dataset_PatientWise
```

Main scripts:

```
main_pipeline.py
segmentation.py
enhancement.py
ldct_simulation.py
cnn_inference.py
metrics_all.py
```

---

## Model

CNN architecture:

```
Input Channels: 3
- LDCT
- Region-Adaptive Output
- Lung Mask

Conv(3 → 32)
ReLU
Conv(32 → 32)
ReLU
Conv(32 → 1)
```

Loss function:

```
L1 Loss + SSIM Loss
```

---

## Evaluation Metrics

Image quality is evaluated using:

* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

Metrics are computed **within the lung region mask** to focus on clinically relevant structures.

---

## Results (Example)

| Method              | PSNR   | SSIM  |
| ------------------- | ------ | ----- |
| LDCT                | ~25–27 | ~0.90 |
| Classical Denoising | ~29–31 | ~0.96 |
| Region-Adaptive     | ~31–33 | ~0.98 |
| CNN Refined         | ~33+   | ~0.99 |

---

## Key Features

* Physics-based LDCT noise simulation
* Segmentation-guided denoising
* Region-adaptive filtering
* Lightweight CNN refinement
* Patient-wise evaluation
