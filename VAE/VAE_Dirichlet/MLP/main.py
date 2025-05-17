import os
import torch
import numpy as np
from mlp_model import MLP
from trainer import Trainer
from data_loader import load_data
from config import get_random_hyperparams

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_nonmal.csv"
results_path = "../VAE/results"
run_file = os.path.join(results_path, "run.npy")

# 获取当前 Run
if os.path.exists(run_file):
    run = int(np.load(run_file)[0])
else:
    raise FileNotFoundError("Run file not found. Ensure VAE training has been completed.")

# 获取随机超参数
params = get_random_hyperparams()
print("Using Hyperparameters:", params)

# 数据加载
train_loader, val_loader = load_data(results_path, run, params['batch_size'], meta_file)

# 模型定义
latent_dim = train_loader.dataset[0][0].shape[0]
model = MLP(latent_dim, params['layer_sizes'], params['dropout'], params['Depth']).to(device)

# 训练
trainer = Trainer(params, device, results_path)
trainer.train_model(model, train_loader, val_loader, run)
