# Region-Guided Adaptive Low-Dose CT Denoising Pipeline

## Overview
This repository contains a preprocessing pipeline for Low-Dose CT (LDCT) image denoising.
The pipeline is designed to work with full-volume DICOM chest CT scans and supports
region-guided denoising using lung segmentation.

## Pipeline Steps
1. Image loading (proxy JPG for validation, DICOM-ready)
2. Intensity / HU normalization
3. Wavelet-based denoising
4. CLAHE-based contrast enhancement
5. Lung segmentation using a pre-trained UNet (integration ready)
6. Region-guided denoising (to be applied after segmentation)

## Current Status
- Preprocessing pipeline (normalization, wavelet, CLAHE): ✅ Implemented
- DICOM volume support: ✅ Structure prepared
- Segmentation: ✅ Pre-trained model identified, integration stub added
- Dataset integration: ⏳ Pending (LDCT DICOM dataset to be provided)

## Segmentation Model
A pre-trained UNet model (`model_100_epoch.pth`) from the Kaggle Chest CT Segmentation
dataset is intended for lung mask generation. The model file is not included in this
repository and will be provided separately.

## Folder Structure
