## Lung Segmentation

This project uses a pre-trained UNet model for lung segmentation obtained from the
Kaggle Chest CT Segmentation dataset.

Model file:
- model_100_epoch.pth

The model is not included in this repository due to size constraints and licensing.
It will be provided separately during execution.

The segmentation model will be used to generate lung masks from LDCT slices,
which will guide region-adaptive denoising in the pipeline.
