import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_DIR = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
results_path = "../results"

# Load and sort image file names 加载并排序图像文件名
all_files_list = [f for f in os.listdir(IMAGE_DIR)]
all_files_list.sort()

# Set seeds and cudnn flags 设置随机种子和cudnn选项
torch.manual_seed(int(time.time()))
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Manage run count 管理运行次数
run_file_path = os.path.join(results_path, 'run.npy')
if not os.path.exists(run_file_path):
    np.save(run_file_path, [0])
Run = np.load(run_file_path, allow_pickle=True)[0]
Run += 1
print("Run:", Run)

# Get randomized hyperparameters 获取随机超参数
params = get_random_hyperparams()
print("Using Hyperparameters:", params)

# Prepare data loaders 准备数据加载器
train_loader, test_loader = vae_data_split(
    IMAGE_DIR, meta_file, all_files_list, 
    params['batch_size'], params['HU_UpperBound'], params['HU_LowerBound'])

# Instantiate model 初始化VAE模型
model = DIR_VAE(params['base'], params['latent_size'], params['alpha_fill_value']).to(device)

# Initialize the Trainer 初始化训练器
trainer = Trainer(params, device, results_path=results_path, model=model)

# Training settings 设置训练参数
epochs = 20
sample_shape = (12, params['latent_size'] * params['base'])

# Start model training 开始模型训练
test_loss, ssim_score = trainer.train_model(model, params['lr'], epochs, sample_shape, train_loader, test_loader)
trainer.plot_results(f'loss_curve_{Run}.png')

# Store test results 存储测试结果
ssim_list, loss_list = [], []
ssim_list.append(ssim_score)
loss_list.append(test_loss)
print('Final Test Loss:', test_loss, 
      'Final SSIM:', ssim_score)         


# --------------- Save latent vectors and compute metrics 保存潜变量并计算指标 ---------------
vae_test_loss = test_loss

images = LoadImages(
    main_dir=IMAGE_DIR + '/', 
    files_list=all_files_list, 
    HU_Upper=params['HU_UpperBound'], 
    HU_Lower=params['HU_LowerBound'])
image_loader = DataLoader(images, params['batch_size'], shuffle=False)
model.eval()

# Define loss functions 定义损失函数
MSE = nn.MSELoss(reduction='mean')
l1_loss = nn.L1Loss(reduction='mean') 

# Initialize lists for evaluation 初始化评估列表
mus, log_vars, reconstructions = [], [], []
SSIM_list, MSE_list, L1_list = [], [], []

# Evaluate only if test loss is valid 仅在测试损失有效时进行评估
if not math.isnan(vae_test_loss):
    with torch.no_grad():
        for batch_idx, data in enumerate(image_loader):
            data = data.float().to(device)
            reconstructions_batch, alpha, dirichlet_sample = model(data)
            
            # Save latent vector 保存潜变量
            for mu in alpha:
                mus.append(mu.tolist())
            
            # Calculate SSIM 计算结构相似性指标
            SSIM_batch = ssim(data, reconstructions_batch, data_range=1, nonnegative_ssim=True)
            SSIM_list.append(np.array(SSIM_batch.cpu()).item())
            
            # Calculate Mean Squared Error 计算均方误差
            MSE_batch = MSE(data, reconstructions_batch)
            MSE_list.append(np.array(MSE_batch.cpu()).item())
            
            # Calculate Mean Absolute Error 计算平均绝对误差
            L1_batch = l1_loss(data, reconstructions_batch)
            L1_list.append(np.array(L1_batch.cpu()).item())
        
    print('Number of latent vectors', len(mus))
    print('Mean Squared Error', np.mean(MSE_list))
    print('Mean Absolute Error', np.mean(L1_list))
    print('Mean SSIM', np.mean(SSIM_list))
    
    # Save latent vectors 保存潜变量
    np.save(os.path.join(results_path, f'latent_vectors_{Run}.npy'), mus)
    
    # Prepare final metrics 准备最终指标
    metrics_list = [ssim_score, test_loss, np.mean(MSE_list), np.mean(L1_list), params]   
    
    # Save VAE info for later use 保存VAE信息供后续使用
    vae_info = {
        "vae_test_loss": vae_test_loss,
        "metrics_list": metrics_list,
        "latent_size": params["latent_size"],
        "base": params["base"]
    }
    vae_info_path = os.path.join(results_path, f"vae_metrics_{Run}.npy")
    np.save(vae_info_path, vae_info, allow_pickle=True)
    print(f"Saved VAE metrics to: {vae_info_path}")