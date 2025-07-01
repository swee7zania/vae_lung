import sys
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from data_split import load_meta_and_images
from config import get_random_hyperparams, get_best_hyperparams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE
from MLP.mlp_model import MLP
from trainer_joint import JointTrainer


class NpyDataset(Dataset):
    def __init__(self, df, HU_Upper, HU_Lower, return_label=False):
        self.paths = df["image_path"].tolist()
        self.HU_Upper = HU_Upper
        self.HU_Lower = HU_Lower
        self.return_label = return_label
        if return_label:
            self.labels = df["label"].values
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.load(self.paths[idx])
        # HU 窗口归一化
        img = np.where(
            (self.HU_Lower <= img) & (img <= self.HU_Upper),
            (img - self.HU_Lower) / (self.HU_Upper - self.HU_Lower),
            img
        )
        img[img < self.HU_Lower] = 0
        img[img > self.HU_Upper] = 1
        img_tensor = self.transform(img)
        if self.return_label:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label
        else:
            return img_tensor


def run_kfold_joint_training(meta_file, image_dir, k_folds, params, epochs, results_path, device):
    os.makedirs(results_path, exist_ok=True)

    splits = load_meta_and_images(meta_file, image_dir, k_folds=k_folds)

    for fold, (train_df, val_df) in enumerate(splits):
        print(f"\n================= Fold {fold} =================")

        # === Step 1: Create Datasets ===
        train_set = NpyDataset(train_df, params["HU_UpperBound"], params["HU_LowerBound"], return_label=True)
        val_set = NpyDataset(val_df, params["HU_UpperBound"], params["HU_LowerBound"], return_label=True)

        train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

        # === Step 2: Initialize Models ===
        vae = VAE(params["base"], params["latent_size"]).to(device)
        mlp = MLP(params["latent_size"], params["base"], params['layer_sizes'], params['dropout'], params['Depth']).to(device)

        # === Step 3: Initialize Joint Trainer ===
        trainer = JointTrainer(params, device, results_path, vae, mlp)

        # === Step 4: Train ===
        train_losses, val_losses, ssim_scores = trainer.train_model(epochs, train_loader, val_loader, fold)

        # === Step 5: Save losses & plot ===
        np.save(os.path.join(results_path, f"joint_train_loss_fold{fold}.npy"), train_losses)
        np.save(os.path.join(results_path, f"joint_val_loss_fold{fold}.npy"), val_losses)
        np.save(os.path.join(results_path, f"joint_ssim_fold{fold}.npy"), ssim_scores)

        trainer.plot_results(train_losses, val_losses, f"joint_loss_curve_fold{fold}.png")
        
        # === Step 6: Save final model checkpoint ===
        torch.save({
            'vae_state_dict': vae.state_dict(),
            'mlp_state_dict': mlp.state_dict(),
            'params': params,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'ssim_scores': ssim_scores
        }, os.path.join(results_path, f"joint_model_fold{fold}.pth"))
        


if __name__ == "__main__":
    meta_file = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
    image_dir = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
    results_path = "../results_joint"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = get_random_hyperparams()
    print(f"Using Hyperparameters:{params}\n")

    k_folds = 3
    epochs = 3

    run_kfold_joint_training(meta_file, image_dir, k_folds, params, epochs, results_path, device)

    print("\n================= Joint Training Done =================")
