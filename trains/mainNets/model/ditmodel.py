import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .. import DiT
from ..DiT.models import DiT,DiTBlock, TimestepEmbedder, modulate
from diffusers import DDPMScheduler, DDIMScheduler
import tqdm

class DiTDiffusionModule(nn.Module):
    def __init__(self,
                 in_channels=32,
                 seq_len=48,
                 hidden_size=64,
                 depth=6,
                 num_heads=16,
                 mlp_ratio=4.0,
                 class_dropout_prob=0.1,
                 learn_sigma=True,
                 cond_dim=32):
        super().__init__()
        self.patch_size = 1 
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        
        # 定义自定义的DiT模型
        class LinearEmbedDiT(DiT):
            def __init__(self, cond_dim, **kwargs):
                super().__init__(**kwargs)
                self.y_embedder = nn.Linear(cond_dim, kwargs['hidden_size'])
                self.x_embedder = nn.Sequential(
                    nn.Linear(kwargs['in_channels'], kwargs['hidden_size']),
                )
                self.unpatchify = nn.Sequential(
                    nn.Linear(kwargs['hidden_size'], kwargs['in_channels'])
                )
                self.pos_embed = nn.Parameter(torch.randn(1, seq_len, kwargs['hidden_size']))

            def forward(self, x, t, cond):
                cond = cond.mean(dim=2)  # [B, C]
                x = self.x_embedder(x) + self.pos_embed  # [B, T, D]
                t = self.t_embedder(t)  # [B, D]
                y = self.y_embedder(cond)  # [B, D]
                c = t + y  # [B, D]
                for block in self.blocks:
                    x = block(x, c)  # [B, T, D]
                x = self.final_layer(x, c)  # [B, T, D]
                x = self.unpatchify(x)  # [B, T, C]
                return x

        self.dit = LinearEmbedDiT(
            input_size=seq_len,
            patch_size=1,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=1000,
            learn_sigma=learn_sigma,
            cond_dim=cond_dim
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        x = x.transpose(1, 2)  # [B, T, C]
        cond = cond  # [B, C, T] 
        x = self.dit(x, t, cond)  # [B, T, C]
        if self.learn_sigma:
            x = x[:, :, :self.in_channels]  # 只保留前半部分
        x = x.transpose(1, 2)  # [B, C, T]
        return x

    def compute_loss(
        self, 
        target_x: torch.Tensor, 
        cond: torch.Tensor, 
        scheduler: DDPMScheduler,
        timesteps: int = 1000
    ):
        """
        使用diffusers库的调度器计算扩散模型的损失函数
        
        Args:
            target_x: [B, C, T] - 目标信号 (ground truth)
            cond: [B, C, T] - 条件信号 (如音频/视频特征)
            scheduler: diffusers库的噪声调度器实例
            timesteps: 扩散过程的总步数
        
        Returns:
            loss: 计算出的MSE损失
            noise_pred: 模型预测的噪声
            true_noise: 实际添加的噪声
        """
        B = target_x.shape[0]
        device = target_x.device
        
        # 1. 随机采样时间步 (t ~ Uniform[0, timesteps-1])
        t = torch.randint(0, timesteps, (B,), device=device).long()
        
        # 2. 生成高斯噪声 ε ~ N(0, I)
        noise = torch.randn_like(target_x)
        
        # 3. 使用diffusers调度器添加噪声
        noisy_x = scheduler.add_noise(target_x, noise, t)
        
        # 4. 模型预测噪声: ε_θ(x_t, t, cond)
        noise_pred = self.forward(noisy_x, t, cond)
        
        # 5. 计算MSE损失: L = ||ε - ε_θ||^2
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    def sample(
        self,
        cond: torch.Tensor,
        scheduler: DDIMScheduler,
        noise: torch.Tensor = None,
        num_inference_steps: int = 50,
        progress_bar: bool = False  # 禁用进度条功能
    ):
        """
        使用diffusers库的调度器进行采样
        
        Args:
            cond: [B, C, T] - 条件信号
            scheduler: diffusers库的采样调度器实例 (如DDIMScheduler)
            noise: 初始噪声 [B, C, T]，若为None则自动生成
            num_inference_steps: 采样步数
            progress_bar: 是否显示进度条 (已禁用)
            
        Returns:
            sample: [B, C, T] - 生成的样本
        """
        device = cond.device
        B, _, T = cond.shape
        
        # 初始化噪声
        if noise is None:
            noise = torch.randn(B, self.in_channels, T, device=device)
        else:
            noise = noise.to(device)
        
        # 设置调度器的采样步数
        scheduler.set_timesteps(num_inference_steps, device=device)
        
        # 初始化为完全噪声
        x = noise
        
        # 迭代去噪
        timesteps = scheduler.timesteps
        
        # 简单的迭代循环，没有进度条
        for t in timesteps:
            # 创建当前时间步的批次
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # 模型预测噪声
            model_output = self.forward(x, t_batch, cond)
            
            # 使用调度器更新样本
            x = scheduler.step(model_output, t, x).prev_sample
        
        return x



# class Diffusion:
#     def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
#         self.T = timesteps
#         self.betas = torch.linspace(beta_start, beta_end, timesteps)  # [T]
#         self.alphas = 1. - self.betas  # [T]
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # [T]

#     def q_sample(self, x_start, t, noise):
#         """
#         x_start: [B, C, T]
#         t: [B] indices of timestep
#         noise: [B, C, T]
#         """
#         device = x_start.device

#         # 把 alphas_cumprod 和计算出的值都转移到 x_start 的设备上
#         sqrt_alpha = self.alphas_cumprod[t].sqrt().to(device).view(-1, 1, 1)  # [B, 1, 1]
#         sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().to(device).view(-1, 1, 1)  # [B, 1, 1]

#         return sqrt_alpha * x_start + sqrt_one_minus * noise


# class DiTDiffusionModule(nn.Module):
#     def __init__(self,
#                  in_channels=32,
#                  seq_len=48,
#                  hidden_size=64,
#                  depth=6,
#                  num_heads=16,
#                  mlp_ratio=4.0,
#                  class_dropout_prob=0.1,
#                  learn_sigma=True,
#                  cond_dim=32):
#         super().__init__()
#         self.patch_size = 1 
#         self.learn_sigma = learn_sigma
#         self.in_channels = in_channels
        

#         class LinearEmbedDiT(DiT):
#             def __init__(self, cond_dim, **kwargs):
#                 super().__init__(**kwargs)
#                 # 用线性映射替换 label embedding
#                 self.y_embedder = nn.Linear(cond_dim, kwargs['hidden_size'])

#                 # 用 1D patch 嵌入器替换 patchify
#                 self.x_embedder = nn.Sequential(
#                     nn.Linear(kwargs['in_channels'], kwargs['hidden_size']),
#                 )
#                 self.unpatchify = nn.Sequential(
#                     nn.Linear(kwargs['hidden_size'], kwargs['in_channels'])
#                 )

#                 # 使用可学习的位置编码（可选）
#                 self.pos_embed = nn.Parameter(torch.randn(1, seq_len, kwargs['hidden_size']))

#             def forward(self, x, t, cond):
#                 # x: [B, T, C]
#                 # cond: [B, C, T]  ← 旧格式，先降维
#                 cond = cond.mean(dim=2)          # [B, C]

#                 # print("x shape:", x.shape)
#                 x = self.x_embedder(x) + self.pos_embed  # [B, T, D]
#                 # print("x shape:", x.shape)

#                 t = self.t_embedder(t)                   # [B, D]
#                 y = self.y_embedder(cond)                # [B, D]
#                 c = t + y                                 # [B, D]
#                 for block in self.blocks:
#                     x = block(x, c)                       # [B, T, D]
#                 x = self.final_layer(x, c)                # [B, T, D]
#                 # print("x shape:", x.shape)
#                 x = self.unpatchify(x)                    # [B, T, C] (C = out_channels)
#                 return x

#         self.dit = LinearEmbedDiT(
#             input_size=seq_len,
#             patch_size=1,
#             in_channels=in_channels,
#             hidden_size=hidden_size,
#             depth=depth,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             class_dropout_prob=class_dropout_prob,
#             num_classes=1000,
#             learn_sigma=learn_sigma,
#             cond_dim=cond_dim
#         )

#     def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
#         """
#         x: [B, C, T]
#         t: [B]
#         cond: [B, C, T] ← 例如 proj_x_a
#         return: [B, C, T]
#         """
#         x = x.transpose(1, 2)                  # [B, T, C]
#         cond = cond                            # [B, C, T] (pass-through)
#         x = self.dit(x, t, cond)               # [B, T, C]
#         if self.learn_sigma:
#             x = x[:, :, :self.in_channels]     # 如果 learn_sigma 开启，只保留前半部分
#         x = x.transpose(1, 2)                  # [B, C, T]
#         return x
    
#     def compute_loss(self, target_x: torch.Tensor, cond: torch.Tensor, diffusion: Diffusion):
#         """
#         target_x: [B, C, T] - 要恢复的模态 (ground truth)
#         cond:     [B, C, T] - 提供的条件模态 (如音频 + 视频)
#         diffusion: Diffusion 实例，提供 q_sample 操作

#         return: (loss, pred, noise)
#         """
#         B = target_x.shape[0]
#         device = target_x.device

#         # Step 1: 随机采样扩散时间步 t ∈ [0, T)
#         t = torch.randint(0, 1000, (B,), device=device)

#         # Step 2: 生成高斯噪声 ε
#         noise = torch.randn_like(target_x)

#         # Step 3: 构造带噪版本的输入 x_t
#         x_noised = diffusion.q_sample(target_x, t, noise)  # [B, C, T]

#         # Step 4: 模型预测噪声
#         noise_pred = self.forward(x_noised, t, cond)       # [B, C, T]

#         # Step 5: 损失函数（MSE between predicted noise and true noise）
#         loss = F.mse_loss(noise_pred, noise)

#         return loss
    
    # def sample(self, cond: torch.Tensor, diffusion: Diffusion, noise: torch.Tensor = None, steps: int = None):
    #     """
    #     cond: [B, C, T]，条件模态（如音频+视觉）
    #     diffusion: Diffusion 实例，提供betas/alphas
    #     noise: 初始高斯噪声 [B, C, T]，若为 None 则自动生成
    #     steps: 采样步数（默认使用 diffusion.T）

    #     return: [B, C, T]，生成的目标模态
    #     """
    #     B, C, T = cond.shape
    #     steps = diffusion.T if steps is None else steps
    #     device = cond.device

    #     if noise is None:
    #         x = torch.randn(B, self.in_channels, T, device=device)
    #     else:
    #         x = noise.to(device)

    #     for i in reversed(range(steps)):
    #         t = torch.full((B,), i, device=device, dtype=torch.long)

    #         # 模型预测噪声 ε_theta(x_t, t, cond)
    #         eps_theta = self.forward(x, t, cond)  # [B, C, T]

    #         # 提取当前的 alpha/beta 并转到 device
    #         beta = diffusion.betas[t].to(device).view(-1, 1, 1)           # [B, 1, 1]
    #         alpha = diffusion.alphas[t].to(device).view(-1, 1, 1)
    #         alpha_hat = diffusion.alphas_cumprod[t].to(device).view(-1, 1, 1)

    #         # 根据 DDPM 反推公式：μ_t = (1/√α_t) * (x_t - β_t / √(1 - α_hat_t) * ε)
    #         coef1 = 1 / alpha.sqrt()
    #         coef2 = beta / (1 - alpha_hat).sqrt()
    #         mean = coef1 * (x - coef2 * eps_theta)

    #         if i > 0:
    #             noise = torch.randn_like(x)
    #             sigma = beta.sqrt()
    #             x = mean + sigma * noise  # 采样 x_{t-1}
    #         else:
    #             x = mean  # 最后一步不加噪声
            
    #         # print("x shape:",x.shape)

    #     return x




