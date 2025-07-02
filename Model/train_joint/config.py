import random

PARAMETER_SPACE = {
    # VAE Hyperparameters
    "HU_UpperBound": [600],
    "HU_LowerBound": [-1000],
    "base": [64],
    "latent_size": [4],
    "batch_size": [128],
    "annealing": [0],
    "alpha": [0.8],
    "beta": [5],
    "lr": [0.0001],

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

# 通过循环 100 次寻找到的最优超参数
def get_best_hyperparams():
    return {
        "HU_UpperBound": 600,
        "HU_LowerBound": -1000,
        "base": 64,
        "latent_size": 4,
        "batch_size": 128,
        "annealing": 0,
        "alpha": 0.8,
        "beta": 5,
        "lr": 0.0001,
        "layer_sizes": [2048, 512, 128],
        "dropout": 0.2,
        "Depth": 4,
        # "threshold": 0.5
    }
