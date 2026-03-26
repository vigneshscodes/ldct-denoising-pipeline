import numpy as np


def simulate_ldct(img_norm,
                  lower,
                  upper,
                  dose_factor=0.15,
                  base_photon_count=1500,
                  gaussian_sigma=0.002):
    """
    Physics-inspired LDCT simulation using Beer-Lambert + Poisson noise.

    Improvements:
    - Exponential attenuation (more realistic than linear)
    - Log transform (approximates CT reconstruction physics)
    - Reduced Gaussian noise (secondary effect)

    Returns:
    - ldct_norm (0–1)
    - ldct_hu (HU scale)
    """

    epsilon = 1e-6

    # ---------------------------------
    # Step 1: Simulate attenuation
    # (Beer-Lambert approximation)
    # ---------------------------------
    attenuation = np.exp(-img_norm)

    # ---------------------------------
    # Step 2: Photon count with dose scaling
    # ---------------------------------
    photon_count = base_photon_count * dose_factor * attenuation
    photon_count = np.clip(photon_count, 1, None)

    # ---------------------------------
    # Step 3: Apply Poisson noise
    # ---------------------------------
    noisy_photons = np.random.poisson(photon_count)

    # ---------------------------------
    # Step 4: Log transform (reconstruction)
    # ---------------------------------
    ldct_norm = -np.log((noisy_photons + epsilon) /
                        (base_photon_count * dose_factor))

    # ---------------------------------
    # Step 5: Normalize to [0,1]
    # ---------------------------------
    ldct_norm = (ldct_norm - ldct_norm.min()) / (
        ldct_norm.max() - ldct_norm.min() + epsilon
    )

    # ---------------------------------
    # Step 6: Add small Gaussian noise
    # ---------------------------------
    gaussian_noise = np.random.normal(0, gaussian_sigma, img_norm.shape)
    ldct_norm = ldct_norm + gaussian_noise
    ldct_norm = np.clip(ldct_norm, 0, 1)

    # ---------------------------------
    # Step 7: Convert back to HU
    # ---------------------------------
    ldct_hu = ldct_norm * (upper - lower) + lower

    return ldct_norm, ldct_hu