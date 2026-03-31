# Low-Dose CT Denoising using Region-Adaptive Filtering and CNN Refinement

## Overview

This project presents a hybrid pipeline for Low-Dose CT (LDCT) denoising, combining classical image processing with a lightweight CNN for refinement.

The objective is to reduce noise while preserving lung structures using:

- Physics-based LDCT simulation  
- Segmentation-guided region processing  
- Region-adaptive denoising  
- CNN-based refinement  

---

## Dataset

CT scans are sourced from the **LIDC-IDRI dataset (TCIA)**.

### Dataset Split (Patient-wise)

- **Training:** Patients 1–20  
- **Validation:** Patients 21–23  
- **Testing:** Patients 24–26  

Patient-wise splitting ensures **no data leakage**.

---

## Pipeline
NDCT → LDCT Simulation → Segmentation → Region-Adaptive Denoising → CNN Refinement → Evaluation


---

## Pipeline Stages

### 1. LDCT Simulation
- Poisson noise-based simulation  
- Signal-dependent realistic noise  
- Converts NDCT to LDCT  

---

### 2. Segmentation
- UNet-based lung segmentation  

Generates:
- Lung mask  
- Bone mask  
- Soft tissue mask  

---

### 3. Region-Adaptive Denoising (Phase 2)

Different filters are applied per region:

- **Lung →** Bilateral + NLM  
- **Bone →** NLM + BM3D  
- **Soft Tissue →** BM3D  

This ensures **structure-aware denoising**.

---

### 4. CNN Refinement (Phase 3)

A lightweight CNN refines the denoised image.

#### Input Channels
- LDCT image  
- Region-adaptive output  
- Lung mask  

#### Architecture
Conv (3 → 32) → ReLU
Conv (32 → 32) → ReLU
Conv (32 → 1)


#### Loss Function
L1 + SSIM


The CNN learns **residual refinement** over the denoised image.

---

## Evaluation

Metrics are computed **only inside the lung region**:

- PSNR (Peak Signal-to-Noise Ratio)  
- SSIM (Structural Similarity Index)  

---

## Final Results

| Method | PSNR | SSIM |
|--------|------|------|
| LDCT   | 25.03 ± 0.35 | 0.9536 ± 0.0229 |
| REGION | 29.25 ± 1.43 | 0.9785 ± 0.0089 |
| CNN    | **32.49 ± 0.60** | **0.9907 ± 0.0029** |

---

## Key Observations

- Region-based denoising significantly improves image quality  
- CNN provides final refinement (~+3 dB PSNR gain)  
- No overfitting due to early stopping  
- Strong generalization on unseen test patients  


## Project Structure

```
CT_Datasets
├── NDCT
├── NDCT_Eval
├── LDCT
├── Segmentation
├── Phase2_Output
├── Phase3_CNN_Refined
└── CNN_Dataset_PatientWise
```---

## Key Features

- Physics-based LDCT simulation  
- Segmentation-guided processing  
- Region-adaptive denoising  
- Lightweight CNN refinement  
- Patient-wise evaluation  
- Strong quantitative improvements  

---

## Conclusion

This hybrid approach effectively combines classical denoising methods with CNN-based refinement.

It achieves **high-quality CT reconstruction** with minimal model complexity and strong generalization performance.