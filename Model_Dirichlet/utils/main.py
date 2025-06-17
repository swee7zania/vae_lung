import sys
import os
import torch
import numpy as np
from data_split import load_meta_and_images
from config import get_random_hyperparams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.train_vae import train_vae
from MLP.train_mlp import train_mlp

def run_kfold_training(meta_file, image_dir, k_folds, params, epochs, results_path, device):
    os.makedirs(results_path, exist_ok=True)

    splits = load_meta_and_images(meta_file, image_dir, k_folds=k_folds)


    for fold, (train_df, val_df) in enumerate(splits):
        print(f"\n================= Fold {fold} =================")

        # === Step 1: Train VAE on raw image paths ===
        print("Training VAE model...")
        latent_train, latent_val = train_vae(train_df, val_df, params, epochs, fold, results_path, device)

        # === Step 2: Train MLP on latent vectors ===
        print("Training MLP classifier...")
        labels_train = train_df["label"].values
        labels_val = val_df["label"].values

        train_mlp(latent_train, labels_train, latent_val, labels_val, params, epochs, fold, results_path, device)

def summarize_kfold_metrics(results_path, k_folds):
    metrics_keys = ["precision_pos", "precision_neg", "recall_pos", "recall_neg", "f1_pos", "f1_neg", "auc"]
    all_metrics = {key: [] for key in metrics_keys}
    fold_scores = []  # 保存每个 fold 的综合评分
    
    print("Score = 0.4 * F1_pos + 0.2 * F1_neg + 0.4 * AUC")
    
    for fold in range(k_folds):
        fold_file = os.path.join(results_path, f"mlp_metrics_fold{fold}.npy")
        metrics = np.load(fold_file, allow_pickle=True).item()
        for key in metrics_keys:
            all_metrics[key].append(metrics[key])
            
        # 自定义模型评分公式
        score = (
            0.4 * metrics["f1_pos"] +
            0.2 * metrics["f1_neg"] +
            0.4 * metrics["auc"]
        )
        fold_scores.append(score)
        print(f"Fold {fold} Score = {score:.4f}")
        
    # 找出最优模型
    best_index = int(np.argmax(fold_scores))
    print(f"\nThe best model is: Fold {best_index}，Score = {fold_scores[best_index]:.4f}")
    
    # 统计均值和标准差
    summary = {}
    print("\n========= K-Fold Metrics Summary =========")
    for key in metrics_keys:
        values = all_metrics[key]
        mean = np.mean(values)
        std = np.std(values)
        summary[key] = {"mean": mean, "std": std}
        print(f"{key:<15} μ = {mean:.4f} | σ = {std:.4f}")

    
    # 保存成 npy 文件
    # summary_path = os.path.join(results_path, "kfold_metrics_summary.npy")
    # np.save(summary_path, summary, allow_pickle=True)
    
    return summary

def select_best_model(summary, weights=None):
    # Set weight
    if weights is None:
        weights = {
            "precision_pos": 0.2,
            "recall_pos": 0.2,
            "f1_pos": 0.2,
            "recall_neg": 0.2,
            "auc": 0.2
        }
    score = 0
    print("\n========= Model Score Breakdown =========")
    for metric, weight in weights.items():
        value = summary.get(metric, {}).get("mean", 0)
        partial_score = weight * value
        score += partial_score
        print(f"{metric:<15} × {weight:.2f} = {partial_score:.4f}")

    print(f"\n🎯 Final Combined Score = {score:.4f}")
    return score


if __name__ == "__main__":
    meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
    image_dir = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
    results_path = "../results"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    params = get_random_hyperparams()
    print(f"Using Hyperparameters:{params}\n")
    
    k_folds = 5
    epochs = 5
    
    # 初始化训练器
    os.makedirs(results_path, exist_ok=True)

    run_kfold_training(meta_file, image_dir, k_folds=k_folds, params=params, epochs=epochs, results_path=results_path, device=device)

    # 计算所有折的 μ / σ
    summary = summarize_kfold_metrics(results_path, k_folds)
        
    print("\n================= Over =================")