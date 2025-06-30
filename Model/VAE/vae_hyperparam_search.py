# hyperparam_search.py
import itertools
import random
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from train_vae import train_vae

# ========== 路径配置 ==========
meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
image_dir = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
results_root = "./vae_search_results"
os.makedirs(results_root, exist_ok=True)

# ========== 数据划分 ==========
meta_df = pd.read_csv(meta_file)
meta_df["image_path"] = meta_df["original_image"].apply(lambda x: os.path.join(image_dir, x + ".npy"))

train_df = meta_df[meta_df["data_split"] == "Train"].reset_index(drop=True)
val_df = meta_df[meta_df["data_split"] == "Validation"].reset_index(drop=True)
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# ========== 超参数空间 ==========
param_space = {
    "HU_UpperBound": [300, 400, 500, 600],
    "HU_LowerBound": [-1000, -800, -700, -500],
    "base": [32, 64],
    "latent_size": [4, 8, 16],
    "annealing": [0, 1],
    "alpha": [0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85],
    "beta": [0.5, 0.8, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 250],
    "lr": [1e-6, 1e-5, 5e-5, 1e-4],
    "batch_size": [64, 128],
}

param_keys = list(param_space.keys())
param_values = list(param_space.values())
all_combinations = list(itertools.product(*param_values))
random.shuffle(all_combinations)

total_combos = min(100, len(all_combinations))
print(f"🚀 总共有 {len(all_combinations)} 种超参数组合，将运行其中的 {total_combos} 次")

# ========== 搜索与记录 ==========
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for combo_id, combo in enumerate(all_combinations[:total_combos]):
    param_dict = dict(zip(param_keys, combo))
    print(f"\n========== Running Combo {combo_id + 1}/{total_combos} ==========")
    print(f"hyperparam: {param_dict}")

    results_path = os.path.join(results_root, f"combo_{combo_id}")
    os.makedirs(results_path, exist_ok=True)

    try:
        latent_train, latent_val = train_vae(
            train_df=train_df,
            val_df=val_df,
            params=param_dict,
            epochs=10,
            fold=combo_id,
            results_path=results_path,
            device=device,
        )

        metrics_file = os.path.join(results_path, f"vae_metrics_fold{combo_id}.npy")
        metrics = dict(np.load(metrics_file, allow_pickle=True).item())
        metrics.update(param_dict)
        metrics["combo_id"] = combo_id
        results.append(metrics)

    except Exception as e:
        print(f"❌ Failed on combo {combo_id}: {e}")
        continue

# ========== 保存为 Excel ==========
df = pd.DataFrame(results)
df.to_excel(os.path.join(results_root, "vae_hyperparam_results.xlsx"), index=False)
print("✅ 搜索完成，结果保存至 Excel。")
