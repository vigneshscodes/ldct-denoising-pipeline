import numpy as np


def simulate_ldct(img_hu,
                  dose_factor=0.25,
                  I0=1e6,
                  gaussian_sigma=0.001):
    """
    Physically consistent LDCT simulation using Beer-Lambert law.

    Input:
    - img_hu: CT image in Hounsfield Units

    Output:
    - ldct_norm (for model use)
    - ldct_hu
    """

    epsilon = 1e-6

    # ---------------------------------
    # Step 1: Convert HU → linear attenuation (mu)
    # ---------------------------------
    mu_water = 0.02
    mu = mu_water * (img_hu / 1000.0 + 1)

    # ---------------------------------
    # Step 2: Beer-Lambert law
    # ---------------------------------
    I = I0 * np.exp(-mu)

    # ---------------------------------
    # Step 3: Dose reduction
    # ---------------------------------
    I_low = I * dose_factor
    I_low = np.clip(I_low, 1, None)

    # ---------------------------------
    # Step 4: Poisson noise
    # ---------------------------------
    noisy_I = np.random.poisson(I_low).astype(np.float32)

    # ---------------------------------
    # Step 5: Log reconstruction
    # ---------------------------------
    mu_noisy = -np.log((noisy_I + epsilon) / (I0 * dose_factor))

    # ---------------------------------
    # Step 6: Convert back to HU
    # ---------------------------------
    ldct_hu = (mu_noisy / mu_water - 1) * 1000

    # ---------------------------------
    # Step 7: Normalize (ONLY for model)
    # ---------------------------------
    ldct_norm = (ldct_hu + 1000) / 1400
    ldct_norm = np.clip(ldct_norm, 0, 1).astype(np.float32)

    # ---------------------------------
    # Step 8: Small electronic noise (CONTROLLED)
    # ---------------------------------
    noise = np.random.normal(0, gaussian_sigma, ldct_norm.shape).astype(np.float32)
    ldct_norm = ldct_norm + noise
    ldct_norm = np.clip(ldct_norm, 0, 1)

    return ldct_norm, ldct_hu