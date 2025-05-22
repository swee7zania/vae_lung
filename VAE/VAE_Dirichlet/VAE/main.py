import os
import time
import torch
import numpy as np
import torch.nn as nn
from data_loader import vae_data_split
from trainer import Trainer
from config import get_random_hyperparams
from dirichlet_vae import DIR_VAE
from torch.utils.data import DataLoader
from data_loader import LoadImages
import math
from pytorch_msssim import ssim

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_nonmal.csv"
results_path = "../results"

all_files_list = [f for f in os.listdir(IMAGE_DIR)]
all_files_list.sort()

# settings for reproducibility
torch.manual_seed(int(time.time()))
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

run_file_path = os.path.join(results_path, 'run.npy')
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
trainer = Trainer(params, device, Run=Run, results_path=results_path, model=model)

# Set learning rate and epochs
epochs = 20
sample_shape = (12, params['latent_size'] * params['base'])

# Start training
test_loss, ssim_score = trainer.train_model(model, params['lr'], epochs, sample_shape, train_loader, test_loader)
trainer.plot_results(f'loss_curve_{Run}.png')

ssim_list, loss_list = [], []
ssim_list.append(ssim_score)
loss_list.append(test_loss)
print('Final Test Loss:', test_loss, 
      'Final SSIM:', ssim_score)         

vae_test_loss = test_loss

# Save potential vectors
images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=all_files_list, HU_Upper=params['HU_UpperBound'], HU_Lower=params['HU_LowerBound'])
image_loader = DataLoader(images, params['batch_size'], shuffle=False)
model.eval()
MSE = nn.MSELoss(reduction='mean')
l1_loss = nn.L1Loss(reduction='mean') 
mus, log_vars, reconstructions = [], [], []
SSIM_list, MSE_list, L1_list = [], [], []
if not math.isnan(vae_test_loss):
    with torch.no_grad():
        for batch_idx, data in enumerate(image_loader):
            data = data.float().to(device)
            reconstructions_batch, alpha, dirichlet_sample = model(data)
            # save latent vectors
            for mu in alpha:
                mus.append(mu.tolist()) #torch.squeeze(torch.squeeze(alpha, dim=1), dim=1).tolist())  
            # calculate SSIM
            SSIM_batch = ssim(data, reconstructions_batch, data_range=1, nonnegative_ssim=True)
            SSIM_list.append(np.array(SSIM_batch.cpu()).item())
            # calculate MSE
            MSE_batch = MSE(data, reconstructions_batch)
            MSE_list.append(np.array(MSE_batch.cpu()).item())
            # calculate MAE
            L1_batch = l1_loss(data, reconstructions_batch)
            L1_list.append(np.array(L1_batch.cpu()).item())

        
    print('Number of latent vectors', len(mus))
    print('Mean Squared Error', np.mean(MSE_list))
    print('Mean Absolute Error', np.mean(L1_list))
    print('Mean SSIM', np.mean(SSIM_list))

    np.save(os.path.join(results_path, f'latent_vectors_{Run}.npy'), mus)
        
    metrics_list = [ssim_score, test_loss, np.mean(MSE_list), np.mean(L1_list), params]   
    
    # Save test_loss and evaluate metrics, used in MLP classifiers
    vae_info = {
        "vae_test_loss": vae_test_loss,
        "metrics_list": metrics_list,
        "latent_size": params["latent_size"],
        "base": params["base"]
    }
    vae_info_path = os.path.join(results_path, f"vae_metrics_{Run}.npy")
    np.save(vae_info_path, vae_info, allow_pickle=True)
    print(f"Saved VAE metrics to: {vae_info_path}")

