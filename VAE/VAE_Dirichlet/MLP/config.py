import random

PARAMETER_SPACE = {
    "threshold":[0.6,0.55,0.5,0.45,0.4],
    "lr":[1e-6,1e-5,1e-4],
    "layer_sizes":[[2048, 2048, 1024], [2048, 1024, 512], [2048, 1024, 256], [2048, 512, 512],
                   [2048, 512, 256], [2048, 512, 128], [1024, 1024, 512], [1024, 1024, 256],
                   [1024, 512, 512], [1024, 512, 256], [1024, 256, 256], [512, 512, 256],
                   [512, 256, 256]],
    "dropout":[0.2,0.4,0.5,0.6],
    "batch_size":[32,64,128,256,512],
    "Depth":[4, 5]
}

def get_random_hyperparams():
    return {k: random.choice(v) for k, v in PARAMETER_SPACE.items()}
