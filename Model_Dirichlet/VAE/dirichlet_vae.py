import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from pytorch_msssim import ssim, ms_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------- Variational Autoencoder with Dirichlet prior 使用Dirichlet先验的变分自编码器 --------------- 
class DIR_VAE(nn.Module):
    def __init__(self, base, latent_size, alpha_fill_value):
        super(DIR_VAE, self).__init__()
        self.base = base                           # 通道基数，用于乘倍生成特征图
        self.latent_size = latent_size             # 潜变量维度（Dirichlet）
        self.alpha_fill_value = alpha_fill_value   # Dirichlet先验α值

        # ===================== Encoder 编码器部分 =====================
        # The encoder maps the input image to a latent representation (Dirichlet logits)
        # 编码器将输入图像压缩成潜变量向量（作为 Dirichlet 的 logits 输入）
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=1, padding=1),             # Input: 1x64x64 → base×64×64
            Conv(base, 2 * base, 3, stride=1, padding=1),      # base→2base, unchanged size
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),  # Downsampling, 64x64 → 32x32
            Conv(2 * base, 2 * base, 3, stride=1, padding=1),  # Unchanged size
            Conv(2 * base, 4 * base, 3, stride=2, padding=1),  # 32x32 → 16x16
            Conv(4 * base, 4 * base, 3, stride=1, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),  # 16x16 → 8x8 → 4x4
            nn.Conv2d(4 * base, 32 * base, 8),                 # 4x4 → 1x1, the channel has been greatly improved
            nn.GELU(),                                         # Activation function, GELU is smoother than ReLU
            nn.Flatten(),                                      # Output size :32*base
            nn.Linear(32 * base, latent_size * base, bias=False),  # Match source code
            nn.BatchNorm1d(latent_size * base, momentum=0.9),  # Consistent batch normalization
            nn.GELU()
        )

        # Latent representation to Dirichlet logits
        # 将潜变量进一步映射为Dirichlet参数(logits)
        self.alpha_fc = nn.Linear(latent_size * base, latent_size * base)

        # ===================== Decoder 解码器部分 =====================
        # The decoder reconstructs the image from a Dirichlet-sampled latent vector
        # 解码器从 Dirichlet 分布采样的潜变量重建图像
        self.decoder = nn.Sequential(
            nn.Linear(latent_size * base, 32 * base, bias=False),  # Latent Variable High → dimensional vector
            nn.BatchNorm1d(32 * base),
            nn.GELU(),
            nn.Unflatten(1,(32*base,1,1)),                         # Expanded feature graph shape: (C,1,1)
            nn.Conv2d(32*base, 32*base, 1),                        # 1x1 → 1x1, keep the channel
            ConvTranspose(32*base, 4*base, 8),                     # Transposed convolution upsampling: 1x1 → 8x8      
            Conv(4*base, 4*base, 3, padding=1),                    # Convolution smoothing feature 
            ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),# Bilinear interpolation: 8x8 → 16x16       
            Conv(4*base, 2*base, 3, padding=1),                    # Descending channel
            ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),# 16x16 → 32x32
            Conv(2*base, base, 3, padding=1),                      # Re-descending channel          
            ConvUpsampling(base, base, 4, stride=2, padding=1),    # 32x32 → 64x64
            nn.Conv2d(base, 1, 3, padding=1),                      # Output single channel image
            nn.Sigmoid()                                           # Ensure pixel values ​​are in [0,1]
        )
    
    # Encode input to latent vector 编码输入图像，生成潜变量向量
    def encode(self, x):
        return self.encoder(x)    
    
    # Decode latent vector to image 解码潜变量，重建图像
    def decode(self, z):
        return self.decoder(z)
    
    # Full forward pass: encode → sample Dirichlet → decode
    # 整体前向传播流程：编码 → 从Dirichlet采样 → 解码重建
    def forward(self, x):
        # Encode the input image as a latent variable representation
        # 编码输入图像为潜变量表示
        x = self.encode(x)
        # Get the current batch size 获取当前batch大小
        batch_size = x.shape[0]
        # Map out Dirichlet logits 映射出Dirichlet logits
        alpha = self.alpha_fc(x)
        
        # Create a Dirichlet sampler and sample from it
        # 创建Dirichlet采样器并从中采样
        resampler = ResampleDir(self.latent_size * self.base, batch_size, self.alpha_fill_value)
        
        # Sample latent variables from the predictive Dirichlet distribution
        # 从预测Dirichlet分布中采样潜变量
        dirichlet_sample = resampler.sample(alpha)
        
        # Reconstruct the image using the decoder 使用解码器重建图像
        recon_x = self.decoder(dirichlet_sample)
        
        # Return the reconstructed image, alpha parameter, sampling vector
        # 返回重建图像，α参数，采样向量
        return recon_x, alpha, dirichlet_sample
    
    # Loss function: recon loss + KL divergence + optional SSIM
    # 损失函数：重建误差 + KL散度 + （可选）结构相似性SSIM
    def loss_function(self, recon_x, x, alpha, epoch, params):
        annealing = params['annealing']              # 是否使用KL退火
        beta = params['beta']                        # KL散度系数
        alpha_scalar = params['alpha']               # L1 与 SSIM 的权重因子
        ssim_indicator = params['ssim_indicator']    # 是否使用SSIM
        ssim_scalar = params['ssim_scalar']          # SSIM放大因子（数值缩放）
        
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
        resampler = ResampleDir(self.latent_size * self.base, batch_size, self.alpha_fill_value)
        kld = resampler.prior_forward(alpha)    # 解析KL散度
        kld = torch.sum(kld)                    # 求和（batch内加总）

        # ===== Calculate Reconstruction Loss 计算重建损失 =====
        l1_loss = nn.L1Loss(reduction='sum')
        recon_loss = l1_loss(recon_x, x) * scale_factor    # 按像素均值归一化
        
        # If SSIM scalar is set to 2, use batch size to dynamically adjust
        # 如果SSIM scalar被设置为2，使用batch size来动态调节
        if ssim_scalar == 2:
            ssim_scalar = batch_size

        # ===== Reconstruction index of hybrid SSIM and L1 混合SSIM与L1的重建指标 =====
        if ssim_indicator == 0:
            recon_mix = recon_loss
        elif ssim_indicator == 1:
            ssim_loss = 1 - ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
            recon_mix = alpha_scalar * recon_loss + (1 - alpha_scalar) * ssim_loss * ssim_scalar
        elif ssim_indicator == 2:
            ssim_loss = 1 - ms_ssim(x, recon_x, data_range=1, win_size=3)
            recon_mix = alpha_scalar * recon_loss + (1 - alpha_scalar) * ssim_loss * ssim_scalar

        # ===== Weighted combination final loss 加权组合最终损失 =====
        beta_norm = (10 * beta * self.latent_size) / (64 * 64 * batch_size)   # 自适应 KL 缩放因子
        beta_vae_loss = recon_mix + beta_norm * (kld - C).abs()               # 总损失 = 重建 + KL偏差
        
        # Only used for analysis, not involved in gradient propagation 仅用于分析，不参与梯度传播
        pure_loss = recon_loss + kld

        # Calculate SSIM score for evaluation 计算SSIM分数用于评估
        ssim_score = ssim(x, recon_x, data_range=1, nonnegative_ssim=True)

        return beta_vae_loss, recon_loss, kld, ssim_score, pure_loss


# --------------- Dirichlet resampling unit 重采样单元 --------------- 
# Used to sample latent variables from Dirichlet distribution and support KL divergence calculation
# 用于从Dirichlet分布中采样潜变量，同时支持KL散度计算
class ResampleDir(nn.Module):
    def __init__(self, latent_dim, batch_size, alpha_fill_value):
        super(ResampleDir, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Create a prior Dirichlet distribution filled with alpha_fill_value
        # 创建一个先验的Dirichlet分布，所有α值填充为alpha_fill_value
        self.alpha_target = torch.full((batch_size, latent_dim), 
                                       fill_value=alpha_fill_value, 
                                       dtype=torch.float, 
                                       device=device)
    
    # Convert logits to valid concentration parameters for Dirichlet
    # 将模型输出的logits转换为合法的Dirichlet浓度参数α
    def concentrations_from_logits(self, logits):
        alpha_c = torch.exp(logits)                           # 指数变换保证正值
        alpha_c = torch.clamp(alpha_c, min=1e-10, max=1e10)   # 避免梯度爆炸或下溢
        alpha_c = torch.log1p(alpha_c)                        # 平滑处理，增强稳定性
        return alpha_c
    
    # Compute analytical KL divergence between predicted and prior Dirichlet
    # 计算预测分布与先验Dirichlet分布之间的KL散度（解析公式）
    def dirichlet_kl_divergence(self, logits, eps=1e-10):
        # Get the predicted α parameter 获取预测α参数
        alpha_c_pred = self.concentrations_from_logits(logits)
        # Prior total α 先验总α
        alpha_0_target = torch.sum(self.alpha_target, axis=-1, keepdims=True)
        # Predicted total α 预测总α
        alpha_0_pred = torch.sum(alpha_c_pred, axis=-1, keepdims=True)
        
        # Gamma: the first part of the KL divergence formula KL散度公式中的第一部分
        term1 = torch.lgamma(alpha_0_target) - torch.lgamma(alpha_0_pred)
        term2 = torch.lgamma(alpha_c_pred + eps) - torch.lgamma(self.alpha_target + eps)
        
        # Digamma：The second part of the KL divergence formula KL散度公式的第二部分
        term3_tmp = torch.digamma(self.alpha_target + eps) - torch.digamma(alpha_0_target + eps)
        term3 = (self.alpha_target - alpha_c_pred) * term3_tmp
        
        # Combine all terms 返回最终KL散度
        result = torch.squeeze(term1 + torch.sum(term2 + term3, keepdims=True, axis=-1))
        return result
    
    # Calculate the KL divergence and return the KL loss value 计算KL散度并返回KL损失值
    def prior_forward(self, logits):
        latent_vector = self.dirichlet_kl_divergence(logits)
        return latent_vector
    
    # Sample from predicted Dirichlet distribution 从预测的Dirichlet分布中采样潜变量
    def sample(self, logits):
        # Obtain predicted α parameters 获取预测α参数
        alpha_pred = self.concentrations_from_logits(logits)
        # Sampling, maintain batch dimension 采样，保持batch维度
        dir_sample = Dirichlet(alpha_pred).rsample()
        # This will remove dimensions of size 1 and will not apply to single images
        # 这个会移除大小为 1 的维度，不适用单张图片
        # dir_sample = torch.squeeze(Dirichlet(alpha_pred).rsample())
        return dir_sample

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