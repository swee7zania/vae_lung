import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from MLP.trainer import Trainer
from MLP.mlp_model import MLP

class LatentDataset(Dataset):
    def __init__(self, latent_vectors, labels):
        self.X = latent_vectors
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x, y
    
def train_mlp(latent_train, labels_train, latent_val, labels_val, params, epochs, fold, results_path, device):
    # Create datasets and loaders
    train_set = LatentDataset(latent_train, labels_train)
    val_set = LatentDataset(latent_val, labels_val)

    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

    # 初始化模型
    model = MLP(params["latent_size"], params["base"], params['layer_sizes'], params['dropout'], params['Depth']).to(device)

    # 初始化训练器
    trainer = Trainer(params, device, results_path, params["latent_size"], params["base"])
    
    
    # ──────────────────────── Training ────────────────────────
    # 训练模型
    statsrec, results, auc = trainer.train_model(
        model, epochs, train_loader, val_loader
    )
    
    # 保存 MLP 模型
    mlp_ckpt_path = os.path.join(results_path, f"mlp_model_fold{fold}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "stats": statsrec,
    }, mlp_ckpt_path)
    # print(f"Saved MLP model to: {mlp_ckpt_path}")
    
    # 保存训练图
    trainer.plot_results(f"mlp_train_curve_fold{fold}.png", model_path=mlp_ckpt_path, epochs=epochs)
    
    
    # ──────────────────────── Evaluation ────────────────────────
    precision_pos, precision_neg, recall_pos, recall_neg, f1_pos, f1_neg = results
        
    metrics = {
        "precision_pos": precision_pos,
        "precision_neg": precision_neg,
        "recall_pos": recall_pos,
        "recall_neg": recall_neg,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "auc": auc,
        "params": params
    }
    
    print(f"\n───── MLP Fold {fold} Evaluation ─────")
    print(f"AUC             : {auc:.4f}")
    print(f"Prec Positive   : {precision_pos:.4f}")
    print(f"Prec Negative   : {precision_neg:.4f}")
    print(f"Recall Positive : {recall_pos:.4f}")
    print(f"Recall Negative : {recall_neg:.4f}")
    print(f"F1 Positive     : {f1_pos:.4f}")
    print(f"F1 Negative     : {f1_neg:.4f}") 
    print("──────────────────────────────────────")

    np.save(os.path.join(results_path, f"mlp_metrics_fold{fold}.npy"), metrics, allow_pickle=True)
