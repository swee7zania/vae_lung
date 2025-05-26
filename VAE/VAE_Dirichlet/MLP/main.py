import os
import math
import torch
import numpy as np
import pandas as pd
from config import get_random_hyperparams
from data_loader import Cross_Validation
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

meta = pd.read_csv(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv")
labels = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/labels.npy")

results_path = "../results"
run_file = os.path.join(results_path, "run.npy")

# ===== Function: Load VAE latent info and metrics =====
# 加载指定 Run 下的潜变量文件、性能指标和模型参数
def load_info(Run, results_path):
    latent_file = os.path.join(results_path, f"latent_vectors_{Run}.npy")
    vae_info_file = os.path.join(results_path, f"vae_metrics_{Run}.npy")
    
    latent_vectors = np.load(latent_file, allow_pickle=True)   # 潜变量
    vae_info = np.load(vae_info_file, allow_pickle=True).item()
    vae_test_loss = vae_info["vae_test_loss"]    # VAE测试损失
    metrics_list = vae_info["metrics_list"]      # VAE结构评估指标
    latent_size = vae_info["latent_size"]        # 潜变量维度
    base = vae_info["base"]                      # 通道基数
    
    return latent_vectors, vae_test_loss, metrics_list, latent_size, base


if __name__ == "__main__":
    # ===== Step 1: Get current Run index 获取当前Run编号 =====
    if os.path.exists(run_file):
        Run = int(np.load(run_file)[0])+1
    else:
        raise FileNotFoundError("Run file not found. Ensure VAE training has been completed.")
    print(f"Current Run: {Run}")
    
    # ===== Step 2: Load latent vectors and related VAE info =====
    # 加载对应Run下的潜变量和模型信息
    latent_vectors, vae_test_loss, metrics_list, latent_size, base = load_info(Run, results_path)
    
    # ===== Step 3: Load random hyperparameters 随机加载一组超参数 =====
    params = get_random_hyperparams()
    print("Using Hyperparameters:", params)
    
    # Set the number of training rounds and folds
    nepochs = 50
    num_folds = 5

    # ===== Step 4: Begin cross-validation only if VAE test loss is valid =====
    if not math.isnan(vae_test_loss):
        # Used to save the final evaluation results 用于保存最终评估结果
        Mal_Ben = [0,0,0,0,0]
        # Get the batch size from the parameters 从参数中获取批量大小
        batch_size = params["batch_size"]
        # Save the results of each fold 保存每折的结果
        CV_loss, CV_accuracy, CV_results, CV_auc = [], [], [], []
        
        for run in range(num_folds):
            # ===== Step 5.1: Split into train/valid/test sets for current fold =====
            # 为当前fold构建训练、验证和测试集加载器
            train_loader, valid_loader, test_loader = Cross_Validation(run, num_folds, meta, latent_vectors, labels, batch_size)
            
            # ===== Step 5.2: Initialize trainer 初始化Trainer对象 =====
            trainer = Trainer(params=params, 
                              device=device, 
                              results_path=results_path, 
                              latent_size=latent_size, 
                              base=base)
            
            # ===== Step 5.3: Train and evaluate the classifier 训练并评估MLP分类器 =====
            loss, accuracy, results, auc = trainer.train_model(
                nepochs, train_loader, valid_loader, test_loader, params, 
                run_index=Run, fold_index=run)
            
            # ===== Step 5.4: Append results 记录当前折的结果 =====
            CV_loss.append(loss)
            CV_accuracy.append(accuracy)
            CV_results.append(results)
            CV_auc.append(auc)

        # ===== Step 6: Compute averages across folds 汇总每一折的指标 =====
        avg_auc = np.mean(CV_auc)             # Average AUC
        avg_loss = np.mean(CV_loss)           # Average loss
        avg_accuracy = np.mean(CV_accuracy)   # Average accuracy
        
        # Custom evaluation averaging method
        avg_results = trainer.average_metrics(CV_results)
        
        # Save results and parameters 保存结果与参数
        Mal_Ben = [avg_auc, avg_loss, avg_accuracy, avg_results, params]
        
        print("Final Results Based on Fixed Hyperparameters:")
        print("AUC:", avg_auc)
        print("Loss:", avg_loss)
        print("Accuracy:", avg_accuracy)
        print("Performance Metrics:", avg_results)