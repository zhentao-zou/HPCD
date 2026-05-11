# HPCD: Hierarchical Physical-Chain Decoupling with Geo-Semantic MoLoRA for All-in-One Multi-modal Remote Sensing Image Restoration

This is the official implementation of our paper:

> **Hierarchical Physical-Chain Decoupling with Geo-Semantic MoLoRA for All-in-One Multi-modal Remote Sensing Image Restoration**
>
> *Authors: (Your Name)*
>
> arXiv: (To be updated)

## Overview

HPCD (Hierarchical Physical-Chain Decoupling) is a two-stage framework for all-in-one remote sensing image restoration. It performs hierarchical analysis of image degradation and applies targeted restoration using conditional diffusion models.

### Framework Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HPCD: Two-Stage Image Restoration                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Degradation Analysis (DepictQA)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Input: Degraded Remote Sensing Image                                │    │
│  │  Model: Qwen2.5-VL-7B-Instruct + LoRA                              │    │
│  │  Output: Degradation Type + Semantic Description                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Image Restoration (AutoDIR)                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Input: Degraded Image + Degradation Type + Semantic Description    │    │
│  │  Model: Latent Diffusion + NAFNet + MoLoRA                          │    │
│  │  Output: Restored High-Quality Image                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Supported Degradation Types

| Type | Description | Model |
|------|-------------|-------|
| Gaussian | Gaussian Noise | LoRA |
| Stripe | Strip Noise | LoRA |
| Jpeg | JPEG Compression Artifacts | LoRA |
| Inpainting | Inpainting Artifacts | LoRA |
| Blur | Motion/Defocus Blur | LoRA |
| Fog | Atmospheric Fog/Haze | LoRA |
| Cloud | Cloud Cover | LoRA |
| Patch | Adversarial Patch | LoRA |
| Mix | Multiple Combined Degradations | Mix Model |

## Environment Setup

### Option 1: Using environment.yaml (Recommended)

```bash
conda env create -f environment.yaml
conda activate hpcd
```

### Option 2: Manual Installation

#### Step 1 Environment (DepictQA)
```bash
conda create -n hpcd_step1 python=3.10
conda activate hpcd_step1
pip install torch torchvision
pip install transformers>=4.37.0
pip install peft accelerate
pip install pillow numpy tqdm
pip install qwen-vl-utils
```

#### Step 2 Environment (AutoDIR)
```bash
conda create -n hpcd_step2 python=3.8
conda activate hpcd_step2
pip install torch==1.13.1 torchvision==0.14.1
pip install einops k-diffusion omegaconf
pip install pillow tqdm scipy
pip install stable-diffusion-studio
conda install -c conda-forge compilers
```

## Directory Structure

```
HPCD/
├── environment.yaml              # Conda 环境配置文件
├── README.md                     # 本文件
│
├── step1_qwen/                   # ========== Stage 1 ==========
│   ├── step1_qwen_analysis.py   # 退化类型分析脚本
│   ├── requirements.txt          # Step 1 依赖
│   └── outputs/                  # Step 1 输出
│       └── qwen_analysis_results.json  # 分析结果
│
├── step2_hpcd/                   # ========== Stage 2 ==========
│   ├── step2_hpcd_inference.py  # 图像复原脚本
│   ├── requirements.txt          # Step 2 依赖
│   │
│   ├── checkpoints/              # 模型权重 (已配置本地路径)
│   │   ├── hpcd_base/           # HPCD Base 模型
│   │   │   └── hpcd_epoch_20.ckpt
│   │   ├── hpcd_lora/           # 各类退化 LoRA 权重
│   │   │   ├── Gaussian/
│   │   │   ├── Stripe/
│   │   │   ├── Jpeg/
│   │   │   ├── Inpainting/
│   │   │   ├── Blur/
│   │   │   ├── Fog/
│   │   │   ├── Cloud/
│   │   │   └── Patch/
│   │   └── hpcd_mix/            # Mix 退化模型
│   │       └── hpcd_epoch_14.ckpt
│   │
│   ├── stable_diffusion/        # Stable Diffusion 代码
│   ├── NAFNet/                  # NAFNet 代码
│   ├── model/                   # 自定义模型
│   │   └── hpcd_model.py
│   ├── clip-vit-large-patch14/  # CLIP 模型 (6.4GB)
│   │
│   ├── LR/                      # 输入退化图片
│   │   └── *.png
│   │
│   └── outputs/                 # Step 2 输出
│       └── hpcd_results/        # 复原结果
│           ├── Cloud/
│           ├── Blur/
│           ├── Fog/
│           └── ...
│
├── checkpoints/                  # (原始权重目录，如需)
│   ├── qwen_lora/               # Qwen LoRA 权重
│   ├── hpcd_base/
│   ├── hpcd_lora/
│   └── hpcd_mix/
│
├── inputs/                       # 输入图片目录
└── outputs/                      # 输出目录
```

## Quick Start

### Step 1: Degradation Analysis

1. 将待处理的图片放入 `inputs/` 目录（或 `step2_hpcd/LR/` 目录）

2. 确保已下载 Qwen2.5-VL-7B-Instruct 模型，并更新脚本中的模型路径

3. 运行退化分析：
```bash
cd step1_qwen
conda activate hpcd_step1  # 或 hpcd
python step1_qwen_analysis.py
```

4. 输出文件：`step1_qwen/outputs/qwen_analysis_results.json`

**JSON 输出格式示例：**
```json
[
  {
    "image": "cloud_001.png",
    "image_path": "./LR/cloud_001.png",
    "degradation_answer": "The most prominent distortion is cloud cover. The restored image will show..."
  },
  {
    "image": "blur_002.png",
    "image_path": "./LR/blur_002.png",
    "degradation_answer": "The most prominent distortion is motion blur..."
  }
]
```

### Step 2: Image Restoration

运行图像复原：
```bash
cd step2_hpcd
conda activate hpcd_step2  # 或 hpcd
python step2_hpcd_inference.py
```

输出结果将保存在 `step2_hpcd/outputs/hpcd_results/` 目录中，按退化类型分组。

## Command Line Arguments

### Step 1 Arguments
```bash
python step1_qwen_analysis.py \
    --input_dir ./inputs/ \           # 输入图片目录
    --output ./outputs/ \             # 输出目录
    --batch_size 4                   # 批次大小
```

### Step 2 Arguments
```bash
python step2_hpcd_inference.py \
    --input_json ../step1_qwen/outputs/qwen_analysis_results.json \
    --resolution 512 \                # 输出分辨率
    --steps 100 \                     # 扩散步数
    --cfg_text 7.5 \                  # 文本 CFG
    --cfg_image 1.0 \                 # 图像 CFG
    --seed 42                         # 随机种子
```

**详细参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-json` | `../step1_qwen/outputs/qwen_analysis_results.json` | Step 1 分析结果 JSON |
| `--resolution` | `512` | 输出图像分辨率 |
| `--steps` | `100` | 扩散模型采样步数 |
| `--cfg-text` | `7.5` | 文本条件引导强度 |
| `--cfg-image` | `1.0` | 图像条件引导强度 |
| `--seed` | `42` | 随机种子（用于复现） |
| `--output` | `./outputs/hpcd_results/` | 输出目录 |

## Model Weights

### Download Links (To be updated)

| Model | Size | Description |
|-------|------|-------------|
| Qwen2.5-VL-7B-Instruct | ~14GB | Base vision-language model |
| HPCD Base | ~3.5GB | Base restoration model |
| HPCD LoRAs | ~500MB | Per-degradation LoRA weights |
| CLIP ViT-L/14 | ~6.4GB | Feature extractor |

### Model Preparation

1. **Qwen2.5-VL-7B-Instruct**
```bash
# Hugging Face
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

2. **HPCD Weights** - 配置在 `step2_hpcd/checkpoints/` 目录

## Hardware Requirements

### Minimum Requirements
- GPU: 16GB VRAM (e.g., RTX 3090, A100)
- RAM: 32GB
- Storage: 50GB

### Recommended
- GPU: 24GB+ VRAM (e.g., A100 40GB, RTX 4090)
- RAM: 64GB
- Storage: 100GB+ (including models)

## Citation

If you find this work helpful for your research, please cite:

```bibtex
@article{hpcd2026,
  title={Hierarchical Physical-Chain Decoupling with Geo-Semantic MoLoRA for All-in-One Multi-modal Remote Sensing Image Restoration},
  author={},
  journal={IEEE TGRS},
  year={2026}
}
```

## Acknowledgements

This project builds upon:
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Vision-Language Model
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Latent Diffusion
- [Autodir](https://github.com/jiangyitong/AutoDIR) -- Autodir
- [NAFNet](https://github.com/megvii-model/NAFNet) - Basic Image Restoration
- [LoRA](https://github.com/microsoft/LoRA) - Parameter-Efficient Fine-Tuning

## License

This project is for research purposes. Please contact the authors for commercial use.

## Contact

For questions and issues, please open an issue or contact the authors.
