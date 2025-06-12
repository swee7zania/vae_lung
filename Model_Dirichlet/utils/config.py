import random

PARAMETER_SPACE = {
    "batch_size": [64, 128, 256, 512],
    "lr": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 5e-3],
    
    # VAE Hyperparameters
    "HU_UpperBound": [400, 500, 600],
    "HU_LowerBound": [-1000, -800, -700],
    "base": [18, 32],
    "latent_size": [4, 8, 16, 32],
    "annealing": [0, 1],
    "ssim_indicator": [0, 1],
    "alpha": [0.5, 0.7, 0.8],
    "beta": [0.8, 1, 2, 5, 10, 20, 30, 50],
    "ssim_scalar": [1, 2],
    "recon_scale_factor": [1, 2, 3],
    "alpha_fill_value": [0.6, 0.9, 0.99, 3],

    # MLP Hyperparameters
    "threshold": [0.6, 0.55, 0.5, 0.45, 0.4],
    "layer_sizes": [
        [2048, 2048, 1024], [2048, 1024, 512], [2048, 1024, 256], [2048, 512, 512],
        [2048, 512, 256], [2048, 512, 128], [1024, 1024, 512], [1024, 1024, 256],
        [1024, 512, 512], [1024, 512, 256], [1024, 256, 256], [512, 512, 256],
        [512, 256, 256]
    ],
    "dropout": [0.2, 0.4, 0.5, 0.6],
    "Depth": [4, 5]
}

def get_random_hyperparams():
    return {k: random.choice(v) for k, v in PARAMETER_SPACE.items()}
