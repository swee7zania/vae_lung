import sys
import os
import torch
import random
import itertools
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE
from MLP.mlp_model import MLP
from trainer_joint import JointTrainer
from main_joint import NpyDataset

def search_space():
    return {
        "threshold": [0.5],
        "layer_sizes": [
            [2048, 2048, 1024], [2048, 1024, 512], [2048, 1024, 256], [2048, 512, 512],
            [2048, 512, 256], [2048, 512, 128], [1024, 1024, 512], [1024, 1024, 256],
            [1024, 512, 512], [1024, 512, 256], [1024, 256, 256], [512, 512, 256],
            [512, 256, 256]
        ],
        "dropout": [0.2, 0.4, 0.5, 0.6],
        "Depth": [4, 5]
    }

def generate_param_grid():
    space = search_space()
    keys, values = zip(*space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def fixed_params():
    return {
        "HU_UpperBound": 600,
        "HU_LowerBound": -1000,
        "base": 64,
        "latent_size": 4,
        "batch_size": 128,
        "annealing": 0,
        "alpha": 0.8,
        "beta": 5,
        "lr": 0.0001,
    }

def run_search(meta_file, image_dir, results_path, device, epochs=5, max_trials=50):
    os.makedirs(results_path, exist_ok=True)
    
    # ========= 标准数据划分 =========
    meta_df = pd.read_csv(meta_file)
    meta_df["image_path"] = meta_df["original_image"].apply(lambda x: os.path.join(image_dir, x + ".npy"))
    train_df = meta_df[meta_df["data_split"] == "Train"].reset_index(drop=True)
    val_df = meta_df[meta_df["data_split"] == "Validation"].reset_index(drop=True)

    param_grid = generate_param_grid()
    sampled = random.sample(param_grid, min(max_trials, len(param_grid)))

    trial_results = []

    for idx, trial_params in enumerate(tqdm(sampled, desc="Hyperparam Search")):
        print(f"\n=== Trial {idx+1}/{len(sampled)}: {trial_params}")
        params = fixed_params()
        params.update(trial_params)

        # === Datasets ===
        train_set = NpyDataset(train_df, params["HU_UpperBound"], params["HU_LowerBound"], return_label=True)
        val_set = NpyDataset(val_df, params["HU_UpperBound"], params["HU_LowerBound"], return_label=True)

        train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

        # === Models ===
        vae = VAE(params["base"], params["latent_size"]).to(device)
        mlp = MLP(params["latent_size"], params["base"], params['layer_sizes'], params['dropout'], params['Depth']).to(device)

        # === Trainer ===
        trainer = JointTrainer(params, device, results_path, vae, mlp)
        trainer.train_model(epochs, train_loader, val_loader, fold=idx)

        # === Metrics ===
        metrics_dict, cm, labels, preds = trainer.evaluate_classification(val_loader)
        metrics_dict.update(trial_params)
        trial_results.append(metrics_dict)

    # === Save All Results ===
    df_result = pd.DataFrame(trial_results)
    df_result.sort_values("AUC", ascending=False, inplace=True)
    df_result.to_csv(os.path.join(results_path, "mlp_hyperparam_results.csv"), index=False)
    print("\n✅ Hyperparameter search completed. Top results saved.")


if __name__ == "__main__":
    meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
    image_dir = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
    results_path = "mlp_search_results/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_search(meta_file, image_dir, results_path, device, epochs=10, max_trials=50)
