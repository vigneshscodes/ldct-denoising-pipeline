import numpy as np
import cv2


# ---------------------------------
# Minimal Preconditioning (SAFE)
# ---------------------------------
def minimal_preconditioning(img_norm):
    """
    Very mild smoothing to stabilize segmentation
    WITHOUT altering noise statistics significantly
    """
    img_smooth = cv2.GaussianBlur(img_norm, (3, 3), 0.3)
    return img_smooth


# ---------------------------------
# FINAL ENHANCEMENT (NON-DESTRUCTIVE)
# ---------------------------------
def enhance_ldct(img_norm, mode="identity"):
    """
    Enhancement stage kept for structural completeness.
    Does NOT modify image in a way that affects denoising.

    Modes:
    - "identity"  → no change (recommended)
    - "minimal"   → very mild smoothing (optional)
    """

    if mode == "identity":
        return img_norm

    elif mode == "minimal":
        return minimal_preconditioning(img_norm)

    else:
        raise ValueError("Mode must be 'identity' or 'minimal'")