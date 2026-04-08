import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import scipy.ndimage as ndi


# ------------------------------
# Load Model
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
# Preprocess
# ------------------------------
def preprocess_for_model(img_norm):

    img_3ch = np.stack([img_norm]*3, axis=-1).astype(np.float32)
    img_resized = cv2.resize(img_3ch, (256, 256))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    return transform(img_resized).unsqueeze(0)


# ------------------------------
# Predict Lung Mask
# ------------------------------
def predict_lung_mask(model, img_norm):

    input_tensor = preprocess_for_model(img_norm)

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.sigmoid(output).squeeze().cpu().numpy()

    lung_prob = probs[0]

    lung_prob = cv2.resize(
        lung_prob,
        (img_norm.shape[1], img_norm.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    lung_mask = postprocess_lung_mask(lung_prob)

    return lung_prob, lung_mask


# ------------------------------
# Postprocess Lung Mask (BALANCED FIX)
# ------------------------------
def postprocess_lung_mask(lung_prob, threshold=0.4):

    lung_mask = (lung_prob > threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    # Fill gaps (important)
    lung_mask = ndi.binary_fill_holes(lung_mask).astype(np.uint8)

    # Mild smoothing (not aggressive)
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, kernel)

    return lung_mask


# ------------------------------
# Body Mask (STABLE)
# ------------------------------
def create_body_mask(img_hu):

    body_mask = (img_hu > -600).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)

    # Keep largest component only
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(body_mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    body_mask = (labels == largest_label).astype(np.uint8)

    return body_mask


# ------------------------------
# Bone Mask (CLEAN + SAFE)
# ------------------------------
def create_bone_mask(img_hu, bone_threshold=300):

    bone_mask = (img_hu > bone_threshold).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel)

    return bone_mask


# ------------------------------
# Soft Tissue Mask (STRICT + STABLE)
# ------------------------------
def create_soft_tissue_mask(img_hu, lung_mask, bone_mask):

    body_mask = create_body_mask(img_hu)

    soft_mask = body_mask.copy()
    soft_mask[lung_mask == 1] = 0
    soft_mask[bone_mask == 1] = 0

    # HU constraint (important)
    soft_mask[(img_hu < -100) | (img_hu > 300)] = 0

    return soft_mask
