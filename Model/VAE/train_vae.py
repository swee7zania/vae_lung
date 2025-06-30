import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from vae_model import VAE
from trainer import Trainer

# 用于加载 .npy 图像
class NpyDataset(Dataset):
    def __init__(self, df, HU_Upper, HU_Lower):
        self.paths = df["image_path"].tolist()
        self.HU_Upper = HU_Upper
        self.HU_Lower = HU_Lower
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.load(self.paths[idx])

        # Apply HU window normalization
        img = np.where(
            (self.HU_Lower <= img) & (img <= self.HU_Upper),
            (img - self.HU_Lower) / (self.HU_Upper - self.HU_Lower),
            img
        )
        img[img < self.HU_Lower] = 0
        img[img > self.HU_Upper] = 1
        return self.transform(img)



# 抽取 Dirichlet latent 向量
def extract_latents(model, dataloader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for data in dataloader:
            data = data.float().to(device)
            _, mu, _ = model(data)  # 这里提取 mu
            mu = mu.view(mu.size(0), -1)  # 展平：[B, C, 1, 1] -> [B, C]
            latents.append(mu.cpu().numpy())
    return np.concatenate(latents, axis=0)

def train_vae(train_df, val_df, params, epochs, fold, results_path, device):
    # 创建DataLoader
    train_set = NpyDataset(train_df, params["HU_UpperBound"], params["HU_LowerBound"])
    val_set = NpyDataset(val_df, params["HU_UpperBound"], params["HU_LowerBound"])
    
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

    # 初始化模型
    model = VAE(params["base"], params["latent_size"]).to(device)

    # 初始化训练器
    trainer = Trainer(params, device, results_path, model)
    
    
    # ──────────────────────── Training ────────────────────────
    # 训练模型
    train_losses, val_losses, ssim_score_list = trainer.train_model(
        model, params["lr"], epochs, train_loader, val_loader
    )
    
    # 保存 VAE 模型
    vae_ckpt_path = os.path.join(results_path, f"vae_model_fold{fold}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, vae_ckpt_path)
    # print(f"Saved VAE model to: {vae_ckpt_path}")
    
    # 保存训练图
    trainer.plot_results(f"loss_curve_fold{fold}.png", model_path=vae_ckpt_path)

    # 获取latent向量
    latent_train = extract_latents(model, train_loader, device)
    latent_val = extract_latents(model, val_loader, device)
    
    
    # ──────────────────────── Evaluation ────────────────────────
    # 评估 SSIM、MSE、L1 指标
    model.eval()
    MSE = nn.MSELoss(reduction='mean')
    L1 = nn.L1Loss(reduction='mean')
    
    mse_list, l1_list, mus = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.float().to(device)
            recon, alpha, _ = model(batch)
            mus.extend(alpha.cpu().numpy())

            mse_list.append(MSE(batch, recon).item())
            l1_list.append(L1(batch, recon).item())
    
    metrics = {
        "mse": float(np.mean(mse_list)),
        "l1": float(np.mean(l1_list)),
        "ssim": float(np.mean(ssim_score_list)),
        "train_loss": float(np.mean(train_losses)),
        "val_loss": float(np.mean(val_losses)),
        "params": params
    }
      
    print(f"\n───── VAE Fold {fold} Evaluation ─────")
    print(f"MSE Mean   : {np.mean(mse_list):.4f}")
    print(f"L1 Mean    : {np.mean(l1_list):.4f}")
    print(f"Train Loss  : {np.mean(train_losses):.4f}")
    print(f"Val Loss  : {np.mean(val_losses):.4f}")
    print(f"SSIM Mean  : {np.mean(ssim_score_list):.4f}")
    print("──────────────────────────────────────")
    
    # np.save(os.path.join(results_path, f"latent_train_fold{fold}.npy"), latent_train)
    # np.save(os.path.join(results_path, f"latent_val_fold{fold}.npy"), latent_val)
    # np.save(os.path.join(results_path, f"latent_alpha_fold{fold}.npy"), mus)
    np.save(os.path.join(results_path, f"vae_metrics_fold{fold}.npy"), metrics, allow_pickle=True)
    
    return latent_train, latent_val
