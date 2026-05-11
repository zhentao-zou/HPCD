"""
HPCD LoRA Training Script
Hierarchical Physical-Chain Decoupling with Geo-Semantic MoLoRA
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from typing import Optional

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms

from peft import LoraConfig, get_peft_model

from stable_diffusion.ldm.util import instantiate_from_config

NAFNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "NAFNet")
sys.path.append(NAFNET_PATH) if NAFNET_PATH not in sys.path else None

DTYPE = torch.bfloat16

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Qwen", "Dataset")
DEFAULT_JSON_FILE = "DepictQA_Single_Degradation_Q1_Answer_Only_final.json"


def check_pytorch_gradients(model: torch.nn.Module) -> None:
    """打印模型参数的梯度状态"""
    print("PyTorch 模型参数的梯度状态:")
    print("-" * 50)
    for name, param in model.named_parameters():
        has_grad = "有梯度" if param.requires_grad else "无梯度"
        print(f"{name:<50}: {has_grad}")
    print("-" * 50)


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """打印模型中可训练参数的数量"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param:.2f}"
    )


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """生成高斯核"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    g_2d = g[:, None] * g[None, :]
    return g_2d[None, None, :, :]


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    sigma: float = 1.5
) -> torch.Tensor:
    """计算 SSIM"""
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same size.")
    
    C = img1.size(1)
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    
    window = gaussian_kernel(window_size, sigma).to(img1.device, img1.dtype)
    window = window.repeat(C, 1, 1, 1)
    padding = window_size // 2
    groups = C
    
    mu1 = F.conv2d(img1, window, padding=padding, groups=groups)
    mu2 = F.conv2d(img2, window, padding=padding, groups=groups)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=groups) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=groups) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=groups) - mu1_mu2
    
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast_structure = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    return (luminance * contrast_structure).mean()


class SSIMLoss(nn.Module):
    """SSIM Loss"""
    def __init__(
        self,
        data_range: float = 2.0,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        sigma: float = 1.5
    ):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(
            img1, img2, self.data_range,
            self.window_size, self.k1, self.k2, self.sigma
        )


def load_model_from_config(config: OmegaConf, ckpt: str) -> torch.nn.Module:
    """从配置文件加载模型"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    
    if "model_state_dict" in pl_sd:
        base_sd = pl_sd["model_state_dict"]
    elif "state_dict" in pl_sd:
        base_sd = pl_sd["state_dict"]
    else:
        base_sd = pl_sd
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(base_sd, strict=False)
    print(f"Model loaded. Missing keys: {len(m)}, Unexpected keys: {len(u)}")
    return model


def get_alphas_bar(schedule: str = "linear", timesteps: int = 1000) -> torch.Tensor:
    """获取 DDPM alpha 累积乘积"""
    betas = torch.linspace(0.0001, 0.02, timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


class HPCDJsonDataset(Dataset):
    """HPCD 数据集"""
    
    def __init__(
        self,
        json_path: str,
        target_image_root: str,
        resolution: int = 512,
        filter_key: Optional[str] = None
    ):
        print(f"Loading data from {json_path}...")
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        if filter_key:
            self.data = [item for item in self.data if filter_key in item['target_image']]
        
        self.target_image_root = target_image_root
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Dataset loaded with {len(self.data)} samples. Resolution: {resolution}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_and_transform_image(self, path: str) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Warning: Could not load image at {path}. Error: {e}")
            return torch.zeros(3, self.resolution, self.resolution)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        target_path_relative = item["target_image"]
        input_image_path = os.path.join(self.target_image_root, target_path_relative)
        gt_image_path = item["GT_image"]
        
        return {
            "input_image": self._load_and_transform_image(input_image_path),
            "gt_image": self._load_and_transform_image(gt_image_path),
            "text": item["answer"],
        }


def train_hpcd(args):
    """HPCD LoRA 训练主函数"""
    config = OmegaConf.load(args.config) if os.path.exists(args.config) else None
    
    json_file = os.path.join(args.data_root, DEFAULT_JSON_FILE)
        dataset = HPCDJsonDataset(
        json_path=json_file,
        target_image_root=args.data_root,
        resolution=args.resolution,
        filter_key=args.filter_key
    )
    print(f"Dataset size: {len(dataset)}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    model = load_model_from_config(config, args.ckpt)
    model.to(args.device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    unet_model = model.model.diffusion_model
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "time_embed.0", "time_embed.2"
        ],
        bias="none",
        modules_to_save=None
    )
    
    model.model.diffusion_model = get_peft_model(unet_model, lora_config)
    
    if hasattr(model.model.diffusion_model, 'use_checkpoint'):
        model.model.diffusion_model.use_checkpoint = False
    
    model.to(args.device).to(DTYPE)
    
    if hasattr(model, 'NAFNet'):
        for param in model.NAFNet.parameters():
            param.requires_grad = True
        print("NAFNet module set to trainable")
    
    timesteps = 1000
    alphas_bar = get_alphas_bar(timesteps=timesteps).to(args.device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    check_pytorch_gradients(model)
    
    running_ld_loss = 0.0
    running_air_loss = 0.0
    ssim_loss = SSIMLoss(data_range=2.0)
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(data_loader):
            optimizer.zero_grad()
            
            I = batch["input_image"].to(args.device).to(DTYPE)
            I_gt = batch["gt_image"].to(args.device).to(DTYPE)
            texts = batch["text"]
            batch_size = I.shape[0]
            
            with torch.no_grad():
                z0 = model.encode_first_stage(I_gt).mode()
                zI = model.encode_first_stage(I).mode()
                
                if hasattr(model, 'model_assessment'):
                    tokens = model.model_assessment.tokenizer(
                        texts, truncation=True, max_length=77,
                        padding="max_length", return_tensors="pt"
                    )['input_ids'].to(args.device)
                    e = model.model_assessment.clipmodel.text_model(
                        input_ids=tokens
                    ).last_hidden_state
                else:
                    e = torch.zeros(batch_size, 77, 768, device=args.device, dtype=DTYPE)
            
            t = torch.randint(0, timesteps, (batch_size,), device=args.device).long()
            sqrt_alpha_bar = alphas_bar[t].sqrt().view(batch_size, 1, 1, 1)
            sqrt_one_minus_alpha_bar = (1. - alphas_bar[t]).sqrt().view(batch_size, 1, 1, 1)
            epsilon = torch.randn_like(z0)
            z_t = sqrt_alpha_bar * z0 + sqrt_one_minus_alpha_bar * epsilon
            
            xc = torch.cat([z_t, zI], dim=1)
            epsilon_theta = model.model.diffusion_model(xc, t, context=e)
            
            loss_ld = F.mse_loss(epsilon_theta, epsilon, reduction="mean")
            loss_ld.backward()
            
            z_tilde = (z_t - sqrt_one_minus_alpha_bar * epsilon_theta.detach()) / sqrt_alpha_bar
            del epsilon_theta, z_t, epsilon, xc, e
            
            with torch.no_grad():
                I_z_tilde = model.decode_first_stage(z_tilde)
            
            recon_stack = torch.cat([I, I_z_tilde], dim=1)
            I_out = model.NAFNet(recon_stack)
            
            loss_air_mse = F.mse_loss(I_out, I_gt, reduction="mean")
            loss_air_ssim = ssim_loss(I_out, I_gt)
            loss_air = loss_air_mse + 0.2 * F.l1_loss(I_out, I_gt) + 0.2 * loss_air_ssim
            weighted_loss_air = args.lambda_air * loss_air
            
            weighted_loss_air.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    args.grad_clip
                )
            
            optimizer.step()
            
            running_ld_loss += loss_ld.item()
            running_air_loss += weighted_loss_air.item()
            
            pbar.set_postfix({
                "LD": f"{loss_ld.item():.4f}",
                "AIR": f"{weighted_loss_air.item():.4f}"
            })
            
            if (batch_idx + 1) % args.log_interval == 0:
                avg_ld = running_ld_loss / args.log_interval
                avg_air = running_air_loss / args.log_interval
                print(f"[Epoch {epoch+1}, Batch {batch_idx+1}] LD: {avg_ld:.4f}, AIR: {avg_air:.4f}")
                running_ld_loss = 0.0
                running_air_loss = 0.0
        
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"hpcd_lora_epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.model.diffusion_model.save_pretrained(save_path)
            torch.save(model.NAFNet.state_dict(), os.path.join(save_path, "nafnet_weights.pt"))
            print(f"Checkpoint saved to {save_path}")
    
    print("Training finished.")


def main():
    parser = ArgumentParser(description="HPCD LoRA Training Script")
    
    parser.add_argument("--data-root", default=DEFAULT_BASE_DIR, type=str)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_lora_mix"), type=str)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "generate.yaml"), type=str)
    parser.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output_1031_short_text", "autodir_epoch_20.ckpt"), type=str)
    parser.add_argument("--filter-key", default="Mix", type=str, help="Filter dataset by target_image key (e.g., 'Mix', 'Cloud', 'Fog')")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--lora-rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--lambda-air", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_hpcd(args)


if __name__ == "__main__":
    main()
