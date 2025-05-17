import random

# Hyperparameter configuration
PARAMETER_SPACE = {
    "HU_UpperBound": [400, 500, 600],
    "HU_LowerBound": [-1000, -800, -700],
    "base": [18, 32],
    "latent_size": [4, 8, 16, 32],
    "annealing": [0, 1],
    "ssim_indicator": [0, 1],
    "batch_size": [64, 128, 256, 512],
    "alpha": [0.5, 0.7, 0.8],
    "beta": [0.8, 1, 2, 5, 10, 20, 30, 50],
    "lr": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 5e-3],
    "ssim_scalar": [1, 2],
    "recon_scale_factor": [1, 2, 3],
    "alpha_fill_value": [0.6, 0.9, 0.99, 3]
}

def get_random_hyperparams():
    hyperparams = {k: random.choice(v) for k, v in PARAMETER_SPACE.items()}
    return hyperparams
