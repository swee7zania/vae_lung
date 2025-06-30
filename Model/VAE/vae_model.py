import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class VAE(nn.Module):
    def __init__(self, base, latent_size):
        super(VAE, self).__init__()
        self.base = base                           # 通道基数，用于乘倍生成特征图
        self.latent_size = latent_size             # 潜变量维度（Dirichlet）
        # output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=1, padding=1),        # (64 - 3 + 2)/1 + 1  = 64
            Conv(base, 2*base, 3, stride=1, padding=1),   # 64
            Conv(2*base, 2*base, 3, stride=2, padding=1), # (64 - 3 + 2)/2 + 1 = 32
            Conv(2*base, 2*base, 3, stride=1, padding=1), # 32
            Conv(2*base, 4*base, 3, stride=2, padding=1), # (32 - 3 + 2)/2 + 1 = 16
            Conv(4*base, 4*base, 3, stride=1, padding=1), # 16
            Conv(4*base, 4*base, 3, stride=2, padding=1), # (16 - 3 + 2)/2 + 1 = 8
            nn.Conv2d(4*base, 32*base, 8),                # (8 - 8 + 0)/1 + 1 = 1
            nn.GELU()
        )
        self.encoder_mu = nn.Conv2d(32*base, latent_size*base, 1) # (1 - 1)/1 + 1 = 1    ## 32*base = 32*4 = 128  ### 1*512 = 512
        self.encoder_logvar = nn.Conv2d(32*base, latent_size*base, 1)

        self.decoder = nn.Sequential(
             nn.Conv2d(latent_size*base, 32*base, 1),               # (1 - 1)/1 + 1 = 1                       ## 32 64
             ConvTranspose(32*base, 4*base, 8),                     # (1-1)*1 + 2*0 + 1(8-1) + 0 + 1  = 8     ## 64 4         
             Conv(4*base, 4*base, 3, padding=1),                    # (8 - 3 + 2)/1 + 1 = 8                   ## 4 4 
             ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),# (8-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 16     ## 4 4         
             Conv(4*base, 2*base, 3, padding=1),                    # (16 - 3 + 2)/1 + 1 = 16                 ## 4 2 
             ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),# (16-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 32    ## 2 2         
             Conv(2*base, base, 3, padding=1),                      # 32                                      ## 2 1  
             ConvUpsampling(base, base, 4, stride=2, padding=1),    # (32-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 64   ## 1 1 
             nn.Conv2d(base, 1, 3, padding=1),                      # 64                                      ## 1 1
             nn.Sigmoid() #nn.Tanh()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) 
        eps = torch.randn_like(std) # mean=0 , std=1
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    # Loss function: recon loss + KL divergence + optional SSIM
    # 损失函数：重建误差 + KL散度 + （可选）结构相似性SSIM
    def loss_function(self, recon_x, x, mu, logvar, epoch, params):
        annealing = params['annealing']              # 是否使用KL退火
        beta = params['beta']                        # KL散度系数
        alpha_scalar = params['alpha']               # L1 与 SSIM 的权重因子
        
        batch_size = x.shape[0]
        scale_factor = 1 / (batch_size * self.base)  # 损失缩放因子，按像素均值计算

        # Define the linear annealing function (used to gradually increase the KL loss weight)
        # 定义线性退火函数（用于逐渐增加KL损失权重）
        def linear_annealing(init, fin, step, annealing_steps):
            if annealing_steps == 0:
                return fin
            delta = fin - init
            annealed = min(init + delta * step / annealing_steps, fin)
            return annealed
        
        # Use the annealing function to obtain the KL divergence constraint value C
        # 使用退火函数获取KL散度约束值C
        if annealing == 1:
            C = linear_annealing(0, 1, epoch, 100)
        else:
            C = 0

        # ===== Calculate KL Divergence 计算KL散度 =====
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld *= scale_factor

        # ===== Calculate Reconstruction Loss 计算重建损失 =====
        l1_loss = nn.L1Loss(reduction='sum')
        recon_loss = l1_loss(recon_x, x) * scale_factor    # 按像素均值归一化


        # ===== Reconstruction index of hybrid SSIM and L1 混合SSIM与L1的重建指标 =====
        ssim_scalar = batch_size
        ssim_loss = 1 - ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
        recon_mix = alpha_scalar * recon_loss + (1 - alpha_scalar) * ssim_loss * ssim_scalar


        # ===== Weighted combination final loss 加权组合最终损失 =====
        beta_norm = (beta * self.latent_size * self.base) / (64 * 64)   # 参考代码的beta_norm（关键修改点）
        beta_vae_loss = recon_mix + beta_norm * (kld - C).abs()               # 总损失 = 重建 + KL偏差
        
        # Only used for analysis, not involved in gradient propagation 仅用于分析，不参与梯度传播
        pure_loss = recon_loss + kld

        # Calculate SSIM score for evaluation 计算SSIM分数用于评估
        ssim_score = ssim(x, recon_x, data_range=1, nonnegative_ssim=True)

        return beta_vae_loss, recon_loss, kld, ssim_score, pure_loss

# Basic convolution block 带GELU+BN的卷积模块
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


# Convolutional Transpose Block for upsampling (decoder) 
# 反卷积模块（上采样）
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# Convolutional Bilinear Upsampling Block    
#https://distill.pub/2016/deconv-checkerboard/
#  the checkerboard could be reduced by replacing transpose convolutions with bilinear upsampling
# 双线性插值上采样 + 卷积（减少棋盘伪影）
class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()
        
        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')    # 双线性插值
        return self.conv(x)