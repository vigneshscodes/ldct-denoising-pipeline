# ============================
# LDCT SIMULATION (FINAL 100/100)
# ============================

import numpy as np


def simulate_ldct(img_norm,
                  lower,
                  upper,
                  dose_factor=0.10,        # 🔥 stronger noise (key fix)
                  base_photon_count=1200, # slightly reduced → more noise
                  gaussian_sigma=0.003):  # balanced detector noise

    """
    Realistic LDCT simulation using signal-dependent Poisson noise.

    Input:
    - img_norm: normalized image [0,1]
    - lower, upper: HU window bounds

    Output:
    - ldct_norm
    - ldct_hu
    """

    # --------------------------
    # Step 1: Photon count
    # --------------------------
    photon_count = img_norm * (base_photon_count * dose_factor)

    # Avoid zero photons
    photon_count = np.clip(photon_count, 1, None)

    # --------------------------
    # Step 2: Poisson noise
    # --------------------------
    noisy_photons = np.random.poisson(photon_count).astype(np.float32)

    # --------------------------
    # Step 3: Normalize back
    # --------------------------
    ldct_norm = noisy_photons / (base_photon_count * dose_factor)

    # --------------------------
    # Step 4: Add detector noise
    # --------------------------
    noise = np.random.normal(0, gaussian_sigma, img_norm.shape).astype(np.float32)
    ldct_norm = ldct_norm + noise

    # Clamp safely
    ldct_norm = np.clip(ldct_norm, 0, 1)

    # --------------------------
    # Step 5: Back to HU
    # --------------------------
    ldct_hu = ldct_norm * (upper - lower) + lower

    return ldct_norm, ldct_hu