import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from sklearn import metrics
import seaborn as sns

class JointTrainer:
    def __init__(self, params, device, results_path, vae_model, mlp_model):
        self.params = params
        self.device = device
        self.results_path = results_path
        self.vae = vae_model.to(device)
        self.mlp = mlp_model.to(device)
        self.sample_shape = (12, params["latent_size"] * params["base"], 1, 1)

        # 联合优化器：优化两个模型的参数
        self.optimizer = optim.Adam(
            list(self.vae.parameters()) + list(self.mlp.parameters()),
            lr=params["lr"]
        )

        # 损失函数
        self.recon_loss_fn = nn.L1Loss()
        self.bce_loss_fn = nn.BCELoss()

    def train_model(self, epochs, train_loader, val_loader, fold):
        train_losses, val_losses = [], []
        ssim_score_list = []
        beta = self.params["beta"]
        
        # MLP loss 权重：更重视分类准确率 → 用大；更重视重建质量，降低分类影响 → 用小 alpha
        mlp_loss_weight = 1.2      
        recon_loss_weight = 0.8   # recon loss 混合权重
        
        # ===== Early Stopping 设置 =====
        best_val_loss = float('inf')
        patience = 1000  # 如果 val_loss 连续 10 次没有提升，就 early stop
        counter = 0
        best_model_state = None
        best_epoch = 0
    
        for epoch in range(1, epochs + 1):
            print(f"[JointTrainer] Epoch {epoch}/{epochs}")
            train_loss, ssim_score = self.train_one_epoch(train_loader, beta, mlp_loss_weight, recon_loss_weight, epoch)
            val_loss = self.validate(val_loader, beta, mlp_loss_weight, recon_loss_weight)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            ssim_score_list.append(ssim_score)
            
            # ===== Early Stopping 检查 =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                counter = 0
                # 保存当前最优模型
                best_model_state = {
                    'vae': self.vae.state_dict(),
                    'mlp': self.mlp.state_dict()
                }
            else:
                counter += 1
                # print(f"[EarlyStopping] No improvement for {counter}/{patience} epochs.")
                if counter >= patience:
                    # print(f"[EarlyStopping] Stopping early at epoch {epoch}.")
                    break
    
        # 恢复最优模型
        if best_model_state is not None:
            self.vae.load_state_dict(best_model_state['vae'])
            self.mlp.load_state_dict(best_model_state['mlp'])

        # 最后一轮后评估分类性能
        metrics_dict, cm, labels, preds = self.evaluate_classification(val_loader)
        self.plot_confusion_matrix(cm, f"joint_confmat_fold{fold}.png")
        self.plot_roc_curve(labels, preds, f"joint_roc_fold{fold}.png")

        print("\n───── JointTrainer Final Evaluation ─────")
        for k, v in metrics_dict.items():
            print(f"{k:<18}: {v:.4f}")
        print("──────────────────────────────────────────")

        np.save(os.path.join(self.results_path, f"joint_metrics_fold{fold}.npy"), metrics_dict, allow_pickle=True)

        # 返回所有训练信息
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "ssim_scores": ssim_score_list,
            "best_model_state": best_model_state,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "metrics": metrics_dict
        }
    
    def train_one_epoch(self, train_loader, beta, mlp_loss_weight, recon_loss_weight, epoch):
        self.vae.train()
        self.mlp.train()

        total_loss = 0
        total_ssim = []

        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.float().to(self.device)
            labels = labels.float().to(self.device)
            labels = labels.view(-1, 1)

            self.optimizer.zero_grad()

            recon, mu, logvar = self.vae(x)
            z = mu.view(mu.size(0), -1)
            preds = self.mlp(z)

            # === 计算各种重建损失 ===
            recon_l1 = self.recon_loss_fn(recon, x)
            ssim_loss = 1 - ssim(recon, x, data_range=1.0, size_average=True)
            
            # 混合重建损失
            recon_mix = recon_loss_weight * recon_l1 + (1 - recon_loss_weight) * ssim_loss

            # === KL 散度 ===
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

            # === annealing 权重 ===
            annealing_weight = self.params["annealing"]

            # === MLP 二分类损失 ===
            mlp_loss = self.bce_loss_fn(preds, labels)

            # === 总损失 ===
            loss = recon_mix + beta * kld * annealing_weight + mlp_loss_weight * mlp_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_ssim.append(ssim(recon, x, data_range=1.0, size_average=True).item())

            if (batch_idx + 1) % 10 == 0:
                print(f"[Batch {batch_idx+1}] Total Loss: {loss.item():.4f}, Recon Loss: {recon_l1.item():.4f}, MLP Loss: {mlp_loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        avg_ssim = np.mean(total_ssim)
        return avg_loss, avg_ssim

    def validate(self, val_loader, beta, mlp_loss_weight, recon_loss_weight):
        self.vae.eval()
        self.mlp.eval()

        total_loss = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.float().to(self.device)
                labels = labels.float().to(self.device)
                labels = labels.view(-1, 1)

                recon, mu, logvar = self.vae(x)
                z = mu.view(mu.size(0), -1)
                preds = self.mlp(z)

                recon_l1 = self.recon_loss_fn(recon, x)
                ssim_loss = 1 - ssim(recon, x, data_range=1.0, size_average=True)

                recon_mix = recon_loss_weight * recon_l1 + (1 - recon_loss_weight) * ssim_loss

                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                annealing_weight = 1.0
                mlp_loss = self.bce_loss_fn(preds, labels)

                loss = recon_mix + beta * kld * annealing_weight + mlp_loss_weight * mlp_loss
                total_loss += loss.item()

        val_avg_loss = total_loss / len(val_loader)
        return val_avg_loss
   
    def evaluate_classification(self, val_loader):
        self.vae.eval()
        self.mlp.eval()
    
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.float().to(self.device)
                labels = labels.float().to(self.device)
    
                _, mu, _ = self.vae(x)
                z = mu.view(mu.size(0), -1)
                preds = self.mlp(z)
    
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    
        preds = np.concatenate(all_preds).squeeze()
        labels = np.concatenate(all_labels).squeeze()
    
        # ========= 寻找 F1 最优的 threshold =========
        thresholds = np.linspace(0.4, 0.9, 10)
        best_f1, best_threshold = 0, 0.5
        for t in thresholds:
            pred_bin = (preds >= t).astype(int)
            f1 = metrics.f1_score(labels, pred_bin)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
    
        print(f"Best threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")
    
        # ========= 用 best_threshold 重新计算 =========
        preds_binary = (preds >= best_threshold).astype(int)
        tp = np.sum((preds_binary == 1) & (labels == 1))
        fp = np.sum((preds_binary == 1) & (labels == 0))
        tn = np.sum((preds_binary == 0) & (labels == 0))
        fn = np.sum((preds_binary == 0) & (labels == 1))
    
        precision_pos = tp / (tp + fp + 1e-8)
        precision_neg = tn / (tn + fn + 1e-8)
        recall_pos = tp / (tp + fn + 1e-8)
        recall_neg = tn / (tn + fp + 1e-8)
    
        f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-8)
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-8)
        auc = metrics.roc_auc_score(labels, preds)
    
        cm = np.array([[tn, fp], [fn, tp]])
    
        metrics_dict = {
            "Threshold": best_threshold,
            "Precision": precision_pos,
            "prec_neg": precision_neg,
            "Recall": recall_pos,
            "recall_neg": recall_neg,
            "F1 Score": f1_pos,
            "f1_neg": f1_neg,
            "AUC": auc
        }
    
        return metrics_dict, cm, labels, preds

    
    def plot_roc_curve(self, labels, preds, filename):
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        auc = metrics.roc_auc_score(labels, preds)
    
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, filename))
        plt.close()


    def plot_confusion_matrix(self, cm, filename):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, filename))
        plt.close()

    def plot_results(self, train_losses, val_losses, filename):
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.set_title("Joint Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(self.results_path, filename)
        plt.savefig(save_path)
        plt.close()