import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from pytorch_msssim import ssim, ms_ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResampleDir(nn.Module):
    def __init__(self, latent_dim, batch_size, alpha_fill_value):
        super(ResampleDir, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.alpha_target = torch.full((batch_size, latent_dim), fill_value=alpha_fill_value, dtype=torch.float, device=device)

    def concentrations_from_logits(self, logits):
        alpha_c = torch.exp(logits)
        alpha_c = torch.clamp(alpha_c, min=1e-10, max=1e10)
        alpha_c = torch.log1p(alpha_c)
        return alpha_c
    
    # 这两个是损失函数用的
    # 计算 KL 散度 (Kullback-Leibler Divergence)，用于度量 alpha_target 和 alpha_c_pred 之间的分布差异
    def dirichlet_kl_divergence(self, logits, eps=1e-10):
        alpha_c_pred = self.concentrations_from_logits(logits)

        alpha_0_target = torch.sum(self.alpha_target, axis=-1, keepdims=True)
        alpha_0_pred = torch.sum(alpha_c_pred, axis=-1, keepdims=True)

        term1 = torch.lgamma(alpha_0_target) - torch.lgamma(alpha_0_pred)
        term2 = torch.lgamma(alpha_c_pred + eps) - torch.lgamma(self.alpha_target + eps)

        term3_tmp = torch.digamma(self.alpha_target + eps) - torch.digamma(alpha_0_target + eps)
        term3 = (self.alpha_target - alpha_c_pred) * term3_tmp

        result = torch.squeeze(term1 + torch.sum(term2 + term3, keepdims=True, axis=-1))
        return result
    
    # 直接计算 KL 散度，并返回 KL 损失值
    def prior_forward(self, logits): # analytical kld loss
        latent_vector = self.dirichlet_kl_divergence(logits)
        return latent_vector

    def sample(self, logits):
        alpha_pred = self.concentrations_from_logits(logits)
        dir_sample = torch.squeeze(Dirichlet(alpha_pred).rsample())
        return dir_sample


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

class DIR_VAE(nn.Module):
    def __init__(self, base, latent_size, alpha_fill_value):
        super(DIR_VAE, self).__init__()
        self.base = base
        self.latent_size = latent_size
        self.alpha_fill_value = alpha_fill_value

        # Encoder
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=1, padding=1),
            Conv(base, 2 * base, 3, stride=1, padding=1),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),
            Conv(2 * base, 2 * base, 3, stride=1, padding=1),
            Conv(2 * base, 4 * base, 3, stride=2, padding=1),
            Conv(4 * base, 4 * base, 3, stride=1, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),
            nn.Conv2d(4 * base, 32 * base, 8),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * base, latent_size * base, bias=False),  # Match source code
            nn.BatchNorm1d(latent_size * base, momentum=0.9),  # Consistent batch normalization
            nn.GELU()
        )

        # Latent representation
        self.alpha_fc = nn.Linear(latent_size * base, latent_size * base)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size * base, 32 * base, bias=False),
            nn.BatchNorm1d(32 * base),
            nn.GELU(),
            nn.Unflatten(1,(32*base,1,1)),
            nn.Conv2d(32*base, 32*base, 1),                       # (1 - 1)/1 + 1 = 1                              ## 32 64
            ConvTranspose(32*base, 4*base, 8),                    # (1-1)*1 + 2*0 + 1(8-1) + 0 + 1  = 8     ## 64 4         
            Conv(4*base, 4*base, 3, padding=1),                   # (8 - 3 + 2)/1 + 1 = 8                          ## 4 4 
            ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),# (8-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 16     ## 4 4         
            Conv(4*base, 2*base, 3, padding=1),                   # (16 - 3 + 2)/1 + 1 = 16                        ## 4 2 
            ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),# (16-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 32    ## 2 2         
            Conv(2*base, base, 3, padding=1),                     # 32                                             ## 2 1  
            ConvUpsampling(base, base, 4, stride=2, padding=1),    # (32-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 64   ## 1 1 
            nn.Conv2d(base, 1, 3, padding=1),                     # 64                                             ## 1 1
            nn.Sigmoid() #nn.Tanh()
        )
    
    def encode(self, x):
        return self.encoder(x)    
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = self.encode(x)
        batch_size = x.shape[0] #128
        alpha = self.alpha_fc(x)
        resampler = ResampleDir(self.latent_size * self.base, batch_size, self.alpha_fill_value)
        dirichlet_sample = resampler.sample(alpha) # This variable that follows a Dirichlet distribution
        recon_x = self.decoder(dirichlet_sample)   # can be interpreted as a probability that the sum is 1)
        return recon_x, alpha, dirichlet_sample
    
    def loss_function(self, recon_x, x, alpha, epoch, params):
        annealing = params['annealing']
        beta = params['beta']
        alpha_scalar = params['alpha']
        ssim_indicator = params['ssim_indicator']
        ssim_scalar = params['ssim_scalar']
        
        batch_size = x.shape[0]
        scale_factor = 1 / (batch_size * self.base)

        # Linear annealing
        def linear_annealing(init, fin, step, annealing_steps):
            if annealing_steps == 0:
                return fin
            delta = fin - init
            annealed = min(init + delta * step / annealing_steps, fin)
            return annealed

        # Annealing factor
        if annealing == 1:
            C = linear_annealing(0, 1, epoch, 100)
        else:
            C = 0

        # KL Divergence
        resampler = ResampleDir(self.latent_size * self.base, batch_size, self.alpha_fill_value)
        kld = resampler.prior_forward(alpha)
        kld = torch.sum(kld)

        # Reconstruction Loss
        l1_loss = nn.L1Loss(reduction='sum')
        recon_loss = l1_loss(recon_x, x) * scale_factor

        if ssim_scalar == 2:
            ssim_scalar = batch_size

        # SSIM Loss
        if ssim_indicator == 0:
            recon_mix = recon_loss
        elif ssim_indicator == 1:
            ssim_loss = 1 - ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
            recon_mix = alpha_scalar * recon_loss + (1 - alpha_scalar) * ssim_loss * ssim_scalar
        elif ssim_indicator == 2:
            ssim_loss = 1 - ms_ssim(x, recon_x, data_range=1, win_size=3)
            recon_mix = alpha_scalar * recon_loss + (1 - alpha_scalar) * ssim_loss * ssim_scalar

        # KL Scaling
        beta_norm = (10 * beta * self.latent_size) / (64 * 64 * batch_size)
        beta_vae_loss = recon_mix + beta_norm * (kld - C).abs()
        pure_loss = recon_loss + kld

        # Debugging outputs
        ssim_score = ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
        ms_ssim_score = ms_ssim(x, recon_x, data_range=1, win_size=3)
        if epoch % 100 == 1 or epoch < 3:
            print('recon loss: {:.4f}, recon mix: {:.4f}, kld: {:.4f}, kld scaled: {:.4f}, SSIM: {:.4f}, MS-SSIM: {:.4f}'.format(
                recon_loss.item(), recon_mix, kld.item(), (kld - C).abs().item(), ssim_score.item(), ms_ssim_score.item()))

        return beta_vae_loss, recon_loss, kld, ssim_score, pure_loss


# Convolutional Transpose Block for upsampling (decoder)
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
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)