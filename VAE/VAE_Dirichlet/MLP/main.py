import os
import torch
import numpy as np
from config import get_random_hyperparams
from test_hyperparams import test_hyperparams
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_nonmal.csv"
results_path = "../VAE/results"
run_file = os.path.join(results_path, "run.npy")

def load_info(Run, results_path):
    latent_file = os.path.join(results_path, f"latent_vectors_{Run}.npy")
    vae_info_file = os.path.join(results_path, f"vae_metrics_{Run}.npy")
    
    latent_vectors = np.load(latent_file, allow_pickle=True)
    vae_info = np.load(vae_info_file, allow_pickle=True).item()
    vae_test_loss = vae_info["vae_test_loss"]
    metrics_list = vae_info["metrics_list"]
    latent_size = vae_info["latent_size"]
    base = vae_info["base"]
    
    return latent_vectors, vae_test_loss, metrics_list, latent_size, base


if __name__ == "__main__":
    # Get the current Run
    if os.path.exists(run_file):
        Run = int(np.load(run_file)[0])+1
    else:
        raise FileNotFoundError("Run file not found. Ensure VAE training has been completed.")
    print(f"Current Run: {Run}")
    
    # 获取 潜在向量
    latent_vectors, vae_test_loss, metrics_list, latent_size, base = load_info(Run, results_path)
    
    # 获取 随机超参数
    params = get_random_hyperparams()
    print("Using Hyperparameters:", params)
    
    nepochs = 50
    num_folds = 5
    
    # 执行
    print("🔎 Testing MLP classifier for malignant vs non-malignant")
    Mal_NonMal, a, Mal_NonMal2, a2 = test_hyperparams(params=params, data=1, nepochs=nepochs, num_folds=num_folds, vae_test_loss=vae_test_loss, latent_size=latent_size, base=base, Run=Run, results_path=results_path, latent_vectors=latent_vectors, device=device)
    print("malignant vs non-malignant AUC",
          'AUC:', Mal_NonMal[0], 'Test Loss:', Mal_NonMal[1], 'Test Accuracy:', Mal_NonMal[2], 
          'Performance Metrics:', Mal_NonMal[3], "Hyperparams", Mal_NonMal[4]) 
    print("malignant vs non-malignant Accuracy",
          'AUC:', Mal_NonMal2[0], 'Test Loss:', Mal_NonMal2[1], 'Test Accuracy:', Mal_NonMal2[2], 
          'Performance Metrics:', Mal_NonMal2[3], "Hyperparams", Mal_NonMal2[4]) 
    
    #print("🔎 Testing MLP classifier for malignant vs benign")
    #b, Mal_Ben, b2, Mal_Ben2 = test_hyperparams(params=params, data=2, nepochs=nepochs, vae_test_loss=vae_test_loss, latent_size=latent_size, base=base, Run=Run, results_path=results_path, latent_vectors=latent_vectors, device=device)
    #print("malignant vs benign AUC",
    #      'AUC:', Mal_Ben[0], 'Test Loss:', Mal_Ben[1], 'Test Accuracy:', Mal_Ben[2], 
    #      'Performance Metrics:', Mal_Ben[3], "Hyperparams", Mal_Ben[4]) 
    #print("malignant vs benign Accuracy",
    #      'AUC:', Mal_Ben2[0], 'Test Loss:', Mal_Ben2[1], 'Test Accuracy:', Mal_Ben2[2], 
    #      'Performance Metrics:', Mal_Ben2[3], "Hyperparams", Mal_Ben2[4]) 

