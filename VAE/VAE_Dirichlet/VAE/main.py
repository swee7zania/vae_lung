import os
import time
import torch
import numpy as np
import pandas as pd
from data_loader import vae_data_split
from trainer import Trainer
from config import get_random_hyperparams
from dirichlet_vae import DIR_VAE
from torch.utils.data import DataLoader

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_nonmal.csv"

all_files_list = [f for f in os.listdir(IMAGE_DIR)]
all_files_list.sort()

# settings for reproducibility
torch.manual_seed(int(time.time()))
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

run_file_path = os.path.join("results", 'run.npy')
if not os.path.exists(run_file_path):
    print("run.npy 文件不存在，正在初始化...")
    np.save(run_file_path, [0])
Run = np.load(run_file_path, allow_pickle=True)[0]
Run += 1
print("Run:", Run)


# Get hyperparameters
params = get_random_hyperparams()
print("Using Hyperparameters:", params)

# Create data loaders
train_loader, test_loader = vae_data_split(IMAGE_DIR, meta_file, all_files_list, params['batch_size'], params['HU_UpperBound'], params['HU_LowerBound'])

# Initialize model
model = DIR_VAE(params['base'], params['latent_size'], params['alpha_fill_value']).to(device)

# Initialize Trainer
trainer = Trainer(params, device, Run=Run, results_path="results")

# Set learning rate and epochs
epochs = 1
sample_shape = (12, params['latent_size'] * params['base'])

# Start training
trainer.train_model(model, params['lr'], epochs, sample_shape, train_loader, test_loader)
trainer.plot_results(f'loss_curve_{Run}.png')
