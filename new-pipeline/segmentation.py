import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms


# ------------------------------
# Load Pretrained Model
# ------------------------------
def load_segmentation_model(model_path):

    model = smp.Unet(
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet",
        classes=3,
        activation=None
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()

    return model


# ------------------------------
# Preprocess for UNet
# ------------------------------
def preprocess_for_model(img_norm):

    img_3ch = np.stack([img_norm] * 3, axis=-1).astype(np.float32)

    img_resized = cv2.resize(img_3ch, (256, 256)).astype(np.float32)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    input_tensor = transform(img_resized).unsqueeze(0)

    return input_tensor


# ------------------------------
# Predict Lung Mask
# ------------------------------
def predict_lung_mask(model, img_norm):

    input_tensor = preprocess_for_model(img_norm)

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.sigmoid(output)
    probs = probs.squeeze().cpu().numpy()

    lung_prob = probs[0]

    lung_prob = cv2.resize(
        lung_prob,
        (img_norm.shape[1], img_norm.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    return lung_prob


# ------------------------------
# Postprocess Lung Mask
# ------------------------------
def postprocess_lung_mask(lung_prob, threshold=0.5):

    lung_mask = (lung_prob > threshold).astype(np.uint8)

    # Fill small holes (important)
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    # Remove noise
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Smooth edges
    lung_mask = cv2.GaussianBlur(lung_mask.astype(np.float32), (5,5), 0)
    lung_mask = (lung_mask > 0.5).astype(np.uint8)

    return lung_mask


# ------------------------------
# Bone Mask (FINAL FIX)
# ------------------------------
def create_bone_mask(img_hu):

    # tighter HU range (critical fix)
    bone_mask = ((img_hu > 400) & (img_hu < 1500)).astype(np.uint8)

    # remove noise
    kernel = np.ones((3,3), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel)

    # remove very small components
    num_labels, labels = cv2.connectedComponents(bone_mask)
    for i in range(1, num_labels):
        if np.sum(labels == i) < 200:
            bone_mask[labels == i] = 0

    return bone_mask


# ------------------------------
# Soft Tissue Mask
# ------------------------------
def create_soft_tissue_mask(body_mask, lung_mask, bone_mask):

    soft_mask = body_mask.copy()

    soft_mask[lung_mask == 1] = 0
    soft_mask[bone_mask == 1] = 0

    return soft_mask