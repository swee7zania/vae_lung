import os
import math
import torch
import numpy as np
import pandas as pd
from config import get_random_hyperparams
#from test_hyperparams import test_hyperparams
from data_loader import Cross_Validation
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results_path = "../results"
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
    
    print("🔎 Testing MLP classifier for malignant vs benign")
    #b, Mal_Ben, b2, Mal_Ben2 = test_hyperparams(params=params, nepochs=nepochs, num_folds=num_folds, vae_test_loss=vae_test_loss, latent_size=latent_size, base=base, Run=Run, results_path=results_path, latent_vectors=latent_vectors, device=device)
    
    Mal_NonMal = [0,0,0,0,0]
    Mal_Ben = [0,0,0,0,0]

    if not math.isnan(vae_test_loss):
        # 去除 ambiguous 样本
        ambiguous = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/latent vectors/ambiguous.npy")
        latent_vectors2 = [torch.tensor(vec) for i, vec in enumerate(latent_vectors) if i not in ambiguous]
        latent_vectors = torch.stack(latent_vectors2)

        # 加载 meta 和标签
        meta = pd.read_csv(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv")
        labels = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/latent vectors/labels3.npy")

        batch_size = params["batch_size"]

        CV_loss, CV_accuracy, CV_results, CV_auc = [], [], [], []
        
        for run in range(num_folds):
            train_loader, valid_loader, test_loader = Cross_Validation(run, num_folds, meta, latent_vectors, labels, batch_size)
            trainer = Trainer(params=params, device=device, results_path=results_path, latent_size=latent_size, base=base)
            loss, accuracy, results, auc = trainer.train_model(nepochs, train_loader, valid_loader, test_loader, params, run_index=Run, fold_index=run)
            CV_loss.append(loss)
            CV_accuracy.append(accuracy)
            CV_results.append(results)
            CV_auc.append(auc)

        # 结果平均与提取
        avg_auc = np.mean(CV_auc)
        avg_loss = np.mean(CV_loss)
        avg_accuracy = np.mean(CV_accuracy)
        avg_results = trainer.average_metrics(CV_results)
        
        Mal_Ben = [avg_auc, avg_loss, avg_accuracy, avg_results, params]
        
        print("Final Results Based on Fixed Hyperparameters:")
        print("AUC:", avg_auc)
        print("Loss:", avg_loss)
        print("Accuracy:", avg_accuracy)
        print("Performance Metrics:", avg_results)

