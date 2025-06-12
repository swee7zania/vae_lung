import sys
import os
import torch
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


if __name__ == "__main__":
    meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
    image_dir = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
    results_path = "../results"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    params = get_random_hyperparams()
    print("Using Hyperparameters:", params)
    
    k_folds = 5
    epochs = 5
    
    # 初始化训练器
    os.makedirs(results_path, exist_ok=True)

    run_kfold_training(meta_file, image_dir, k_folds=k_folds, params=params, epochs=epochs, results_path=results_path, device=device)

    print("\n================= Over =================")