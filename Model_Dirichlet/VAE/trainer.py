import os
import math
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Trainer class handles model training, validation, early stopping, and result saving
# Trainer类负责模型的训练、验证、早停和结果保存
class Trainer:
    def __init__(self, params, device, results_path, model):
        self.params = params    # Store hyperparameters 存储超参数
        self.device = device    # Set computation device 设置计算设备
        self.results_path = results_path    # Path to save results 保存结果路径
        self.model = model      # VAE model 实例化的VAE模型
    
    
    # --------------- Main training function 主训练函数 --------------- 
    def train_model(self, model, lr, epochs, sample_shape, train_loader, val_loader):
        # Store loss and SSIM scores 存储损失和SSIM
        train_losses, val_losses, ssim_score_list = [], [], []
        # Use Adam optimizer 使用Adam优化器
        optimiser = optim.Adam(model.parameters(), lr=lr)
        # Reduce LR if validation loss plateaus 学习率调度器：验证损失不下降则降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                               threshold=0.001, threshold_mode='abs')
        # For early stopping 早停计数器
        counter = 0
        
        # Train model over several epochs 在多个epoch上训练模型
        for epoch in range(1, epochs + 1):
            # Train for one epoch and evaluate 训练一个epoch并评估
            train_loss, ssim_score, kld = self.train(model, epoch, epochs, optimiser, sample_shape, train_loader)
            val_loss, val_ssim = self.validate(model, epoch, val_loader)
            
            # Update LR based on train loss 根据训练损失更新学习率
            scheduler.step(train_loss)
            counter = self.early_stopping(counter, train_loss, val_loss, min_delta=1)
            
            # If the training is stopped early due to continuous overfitting or loss is NaN
            # 连续过拟合或损失为NaN则提前停止训练
            if counter > 25:
                print('[VAE Trainer] Early stopping triggered at epoch:', epoch)
                break

            if math.isnan(train_loss):
                print('[VAE Trainer] Training stopped due to infinite loss')
                break
            
            # Record metrics 记录训练指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            ssim_score_list.append(ssim_score)
        
        # Save the model and training history after multiple epoch training
        # 保存多个epoch训练后模型及训练历史
        # torch.save({"state_dict": model.state_dict(), "train_losses": train_losses, "val_losses": val_losses, "params": self.params}, self.results_path + '/VAE_params.pt')
        return train_losses, val_losses, ssim_score_list
    
    
    # --------------- Single epoch training function 单个epoch的训练过程 --------------- 
    def train(self, model, epoch, epochs, optimiser, sample_shape, train_loader):
        # Set model to training mode 启用训练模式
        model.train()
        
        # Initialize accumulators 初始化损失和SSIM的累加器
        train_loss, beta_train_loss = 0, 0
        ssim_list = []
        
        # Iterate over training batches 遍历训练数据的每个小批量
        for batch_idx, data in enumerate(train_loader):
            
            data = data.float().to(self.device)
            
            # Clear previous gradients 清除上一步残留梯度，防止累加
            optimiser.zero_grad()
            
            # ======== Forward Pass ========
            # Pass input through model to get output and latent variables
            # 将输入图像送入模型，获得重建图像、Dirichlet参数alpha、采样结果
            recon_batch, alpha, dirichlet_sample = model(data)
            
            # ======== Loss Calculation ========
            # Compute full and partial losses
            # 计算总损失（loss）、重建损失（recon_loss）、KL散度（kld）、
            # SSIM结构相似度指标（ssim_score）以及只考虑重建的纯损失（pure_loss）
            loss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(
                recon_batch, data, alpha, epoch, self.params)
            
            # Save SSIM score for this batch 记录本批次的SSIM值（用于平均）
            ssim_list.append(ssim_score.item())
            
            # ======== Backward Pass ========
            # Compute gradients via backpropagation 反向传播计算梯度
            loss.backward()
            # Update model parameters based on gradients 使用优化器更新参数
            optimiser.step()
            
            # ======== Record loss values ========
            # Accumulate losses for reporting later
            # 将当前batch的纯损失和带beta加权的总损失分别累加，用于后续计算平均值
            train_loss += pure_loss.item()
            beta_train_loss += loss.item()
            
            # ======== Status Output ========
            # Print training progress every 50 batches
            # 每处理100个batch，打印一次当前训练轮的进度和损失值
            if batch_idx % 100 == 0:
                print('[VAE Trainer] Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    pure_loss.item(), loss.item()))
                
            # Stop training early if loss becomes NaN
            # 如果当前损失为NaN，可能出现数值不稳定，立即中止当前epoch
            if math.isnan(loss):
                break
        
        # Visualize real, reconstructed, and generated images 可视化真实图像、重建图像和合成图像
        if((epoch%50==1) or (epoch < 5) or (epoch==epochs-1)):
            # print('12 Real Images')
            img_grid = make_grid(data[:12], nrow=4, padding=12, pad_value=-1)
            plt.figure(figsize=(10,5))
            plt.imshow(img_grid[0].detach().cpu())
            plt.axis('off')
            plt.savefig(self.results_path + "/" + "visualise_real" + str(epoch) + '.png')
            plt.close()
            # plt.show()

            # print('12 Reconstructed Images')
            img_grid = make_grid(recon_batch[:12], nrow=4, padding=12, pad_value=-1)
            plt.figure(figsize=(10,5))
            plt.imshow(img_grid[0].detach().cpu())
            plt.axis('off')
            plt.savefig(self.results_path + "/" + "visualise_reconstructed" + str(epoch) + '.png')
            plt.close()
            # plt.show()

            # print('12 Synthetic Images')
            sample = torch.randn(sample_shape).to(self.device)
            recon_rand_sample = model.decode(sample)
            img_grid = make_grid(recon_rand_sample[:12], nrow=4, padding=12, pad_value=-1)
            plt.imshow(img_grid[0].detach().cpu())
            plt.axis('off')
            plt.savefig(self.results_path + "/" + "visualise_synthetic" + str(epoch) + '.png')
            plt.close()
            # plt.show()
        
        # Compute average losses 计算平均损失
        train_loss /= len(train_loader.dataset)
        beta_train_loss /= len(train_loader.dataset)
        # Compute mean SSIM 计算平均SSIM
        ssim_mean = np.mean(ssim_list)
        
        # print('====> Epoch {}: Average Train Loss: {:.4f}'.format(epoch, train_loss))
        # print('====> Average Beta Train Loss: {:.4f}'.format(beta_train_loss))
        # print('====> Average Train SSIM: {:.4f}'.format(ssim_mean))

        return train_loss, ssim_mean, kld
    # Model evaluation function 模型评估函数
    
    
    # --------------- Single epoch evaluation function 单个epoch的评估过程 --------------- 
    def validate(self, model, epoch, val_loader):
        # Set model to evaluation mode 启用评估模式
        model.eval()
        # Initialize accumulators 初始化损失和SSIM的累加器
        val_loss, beta_val_loss = 0, 0
        ssim_list = []
        
        # Disable gradient calculation for evaluation
        # 在评估模式下禁用梯度计算以节省显存并提升推理效率
        with torch.no_grad():
            # Loop through val data 遍历整个测试集
            for i, data in enumerate(val_loader):
                data = data.float().to(self.device)
                
                # ======== Forward Pass ========
                # Get the reconstruction image, Dirichlet parameters, and sampling vector
                # 获得重建图、Dirichlet参数、采样向量
                recon_batch, alpha, dirichlet_sample = model(data)
                
                # ======== Loss Calculation ========
                # Compute full and partial losses
                # 计算总损失（loss）、重建损失（recon_loss）、KL散度（kld）、
                # SSIM结构相似度指标（ssim_score）以及只考虑重建的纯损失（pure_loss）
                valloss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(
                    recon_batch, data, alpha, epoch, self.params)
                
                # Accumulate loss and SSIM 累加损失与SSIM
                val_loss += pure_loss.item()           # 纯重建误差
                beta_val_loss += valloss.item()       # 包含KL散度的误差
                ssim_list.append(ssim_score.item())     # 当前batch的SSIM分数
                
                # Stop evaluation if numerical instability occurs
                # 若损失为NaN，可能存在梯度爆炸或数值不稳定，立即中断测试
                if math.isnan(valloss):
                    break
        
        # Normalize loss by total number of samples
        # 将累加的总损失除以测试样本数量，得到平均损失
        val_loss /= len(val_loader.dataset)
        beta_val_loss /= len(val_loader.dataset)
        ssim_mean = np.mean(ssim_list)
        
        # Print evaluation results 打印评估指标
        # print('====> Pure Val Loss: {:.4f}'.format(val_loss))
        # print('====> Beta Val Loss: {:.4f}'.format(beta_val_loss))
        # print('====> Average Val SSIM: {:.4f}'.format(ssim_mean))
        
        return val_loss, ssim_mean
    
    # Early stopping logic 简单早停机制
    def early_stopping(self, counter, train_loss, val_loss, min_delta):
        # 如果测试损失明显高于训练损失
        # If the val loss is significantly higher than the training loss
        if (val_loss - train_loss) > min_delta:
            counter += 1
            if counter % 5 == 0:
                print('Early Stopping Counter At:', counter)  
        return counter
    
    # Plot training and val loss curves 绘制训练和测试损失曲线
    def plot_results(self, filename, model_path):  
        # Skip if model file not found 如果模型文件不存在，则跳过绘图
        if not os.path.exists(model_path):
            print(f"[VAE Trainer] Data file {model_path} not found. Skipping plot generation.")
            return

        data = torch.load(model_path)
        loss = data["train_losses"]
        val_loss = data["val_losses"]
        
        # Remove first epoch 去除第一轮数据
        loss = loss[1:]
        val_loss = val_loss[1:]
        
        fig, ax1 = plt.subplots()
        plt.plot(loss, 'm', label = 'train loss')    # 训练损失曲线
        plt.plot(val_loss, 'g', label = 'val loss')    # 测试损失曲线
        plt.yscale("log")                               # 使用对数坐标轴
        plt.legend(loc='lower right')                   # 图例位置
        plt.xlabel('epoch')                             # 横轴：轮数
        plt.ylabel('loss')                              # 纵轴：损失
        plt.title('Training and validation loss')       # 图标题
        fig.savefig(os.path.join(self.results_path, filename))
        plt.show()
        plt.close()