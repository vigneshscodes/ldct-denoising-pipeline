# ldct_simulation.py

import numpy as np


def simulate_ldct(img_norm, lower, upper,
                  dose_factor=0.25,
                  gaussian_sigma=0.01):

    # Scale intensity to simulate dose reduction
    scaled = img_norm * dose_factor

    # Poisson noise (photon noise)
    poisson = np.random.poisson(scaled * 1000) / 1000.0

    # Gaussian detector noise
    gaussian = np.random.normal(0, gaussian_sigma, img_norm.shape)

    ldct_norm = poisson + gaussian
    ldct_norm = np.clip(ldct_norm, 0, 1)

    # Convert back to HU scale
    ldct_hu = ldct_norm * (upper - lower) + lower

    return ldct_norm, ldct_hu