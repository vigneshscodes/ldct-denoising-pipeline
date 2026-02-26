# ldct_simulation.py

import numpy as np


def simulate_ldct(img_norm,
                  lower,
                  upper,
                  dose_factor=0.25,
                  base_photon_count=1500,
                  gaussian_sigma=0.003):
    """
    Realistic LDCT simulation using signal-dependent Poisson noise.

    Parameters:
    - img_norm: windowed image normalized to [0,1]
    - lower, upper: HU window bounds
    - dose_factor: relative dose level (0.25 = 25% dose)
    - base_photon_count: controls noise magnitude
    - gaussian_sigma: small detector noise

    Returns:
    - ldct_norm (0–1)
    - ldct_hu (HU scale)
    """

    # Photon count proportional to signal and dose
    photon_count = img_norm * (base_photon_count * dose_factor)

    # Prevent zero photon regions
    photon_count = np.clip(photon_count, 1, None)

    # Apply Poisson noise
    noisy_photons = np.random.poisson(photon_count)

    # Normalize back to [0,1]
    ldct_norm = noisy_photons / (base_photon_count * dose_factor)

    # Add small Gaussian detector noise
    gaussian_noise = np.random.normal(0, gaussian_sigma, img_norm.shape)

    ldct_norm = ldct_norm + gaussian_noise
    ldct_norm = np.clip(ldct_norm, 0, 1)

    # Convert back to HU
    ldct_hu = ldct_norm * (upper - lower) + lower

    return ldct_norm, ldct_hu