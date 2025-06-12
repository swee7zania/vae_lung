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
    precision, recall, specificity, f1 = results
        
    metrics = {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "params": params
    }
    
    print(f"\n───── MLP Fold {fold} Evaluation ─────")
    print(f"AUC         : {auc:.4f}")           # 显示 ROC 曲线下的面积
    print(f"Precision   : {precision:.4f}")     # 精确率，表示被预测为正的样本中有多少是真正的正类
    print(f"Recall      : {recall:.4f}")        # 召回率，表示所有真正的正类中被模型识别出来的比例
    print(f"Specificity : {specificity:.4f}")   # 特异度，表示所有负样本中被正确识别为负类的比例
    print(f"F1 score    : {f1:.4f}")            # F1 分数是 precision 和 recall 的调和平均
    print("──────────────────────────────────────")

    np.save(os.path.join(results_path, f"mlp_metrics_fold{fold}.npy"), metrics, allow_pickle=True)
