"""
Step 2: 基于 Qwen 分析结果，使用 HPCD 进行图像复原
"""
import os
import sys
import json
import random
from argparse import ArgumentParser
from collections import defaultdict

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 添加本地库路径
sys.path.insert(0, os.path.join(SCRIPT_DIR, "stable_diffusion"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "NAFNet"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "model"))
sys.path.insert(0, SCRIPT_DIR)

import einops
import k_diffusion as K
import torch
import torch.nn as nn
from torch import autocast
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ldm.util import instantiate_from_config
from wavelet_color_fix import adaptive_instance_normalization
from peft import PeftModel

# 配置 (路径相对于 step2_hpcd 目录)
INPUT_JSON = "../step1_qwen/outputs/qwen_analysis_results.json"
BASE_CKPT  = "./checkpoints/hpcd_base/hpcd_epoch_20.ckpt"
CONFIG     = "./checkpoints/generate.yaml"
OUTPUT_DIR = "./outputs/hpcd_results/"

# LoRA 映射
LORA_MAP = {
    "Gaussian":   "./checkpoints/hpcd_lora/Gaussian/",
    "Stripe":     "./checkpoints/hpcd_lora/Stripe/",
    "Jpeg":       "./checkpoints/hpcd_lora/Jpeg/",
    "Inpainting": "./checkpoints/hpcd_lora/Inpainting/",
    "Blur":       "./checkpoints/hpcd_lora/Blur/",
    "Fog":        "./checkpoints/hpcd_lora/Fog/",
    "Cloud":      "./checkpoints/hpcd_lora/Cloud/",
    "Patch":      "./checkpoints/hpcd_lora/Patch/",
    "Mix":        "./checkpoints/hpcd_mix/hpcd_epoch_14.ckpt",
}

# Mix LoRA 使用的 Base CKPT
MIX_BASE_CKPT = "./checkpoints/hpcd_base/hpcd_epoch_20.ckpt"


def is_mix_degradation(text):
    """检查文本是否包含多个退化类型（复数形式）"""
    return "distortions are" in text.lower()


def detect_single_degradation_type(text):
    """检测文本中包含的单个退化类型（单数形式），返回类型或 None"""
    text_lower = text.lower()
    
    # 单个退化类型映射
    SINGLE_TYPE_PATTERNS = {
        "Gaussian":   ["gaussian noise", "gaussian"],
        "Stripe":    ["strip noise", "stripe"],
        "Jpeg":      ["jpeg compression", "jpeg"],
        "Inpainting":["inpainting"],
        "Blur":      ["blur", "motion blur", "blurry"],
        "Fog":       ["fogging"],
        "Cloud":     ["cloud cover"],
        "Patch":     ["adversarial patch"],
    }
    
    for deg_type, patterns in SINGLE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return deg_type
    
    return None


def classify_degradation(text):
    """根据 Qwen 返回文本分类退化类型
    
    - "The most prominent distortions are..." (复数) → Mix (不使用 LoRA)
    - "The most prominent distortion is..." (单数) → 对应单个退化类型 (使用 LoRA)
    - 无法识别 → 默认 Gaussian
    """
    text_lower = text.lower()
    
    # 检查是否是 Mix (多个退化类型)
    if is_mix_degradation(text):
        return "Mix"
    
    # 检查单个退化类型
    detected_type = detect_single_degradation_type(text)
    if detected_type:
        return detected_type
    
    # 无法识别，默认 Gaussian
    return "Gaussian"


def get_all_degradation_types(text):
    """返回文本中检测到的所有退化类型"""
    return detect_degradation_types(text)

def process_customize_text(response, class_id):
    """
    处理 customize text，支持两种格式：
    1. Qwen 原始格式：开头含 "The prominent distortion varies..." + "The image" 后缀
    2. DepictQA GT 格式："The most prominent distortion is XXX. The restored image will..."
    """
    # DepictQA GT 格式：直接截断 "The restored image will" 之前
    target_phrase = "The restored image will"
    if target_phrase in response:
        customize = response[:response.index(target_phrase)] + "The restored image will depict"
        return customize

    # Qwen 原始格式处理
    if 'Fog' in class_id:
        response = response.replace(
            'The prominent distortion varies, primarily including cloud cover, fog, haze, or other weather-related visual degradation.',
            'The most prominent distortion is fogging.'
        )
    elif class_id == 'Cloud':
        response = response.replace(
            'The prominent distortion varies, primarily including cloud cover, fog, haze, or other weather-related visual degradation.',
            'The most prominent distortion is cloud cover.'
        )
    target_phrase = "The image"
    if target_phrase in response:
        response = response[:response.index(target_phrase)]
    return response


# ===================== 模型工具 =====================
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z     = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond  = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat":    [torch.cat([cond["c_concat"][0],    cond["c_concat"][0],       uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_base_model(config, base_ckpt_path):
    model = instantiate_from_config(config.model)
    model.eval()
    print(f"Loading BASE model from {base_ckpt_path}")
    pl_sd = torch.load(base_ckpt_path, map_location="cpu")
    if "model_state_dict" in pl_sd:
        base_sd = pl_sd["model_state_dict"]
    elif "state_dict" in pl_sd:
        base_sd = pl_sd["state_dict"]
    else:
        base_sd = pl_sd
    model.load_state_dict(base_sd, strict=False)
    return model


def attach_lora(base_model, lora_ckpt_path):
    print(f"Attaching LoRA from {lora_ckpt_path}")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_ckpt_path, is_trainable=False)
    nafnet_ckpt = os.path.join(lora_ckpt_path, "nafnet_weights.pt")
    if os.path.exists(nafnet_ckpt):
        print(f"Loading NAFNet weights from {nafnet_ckpt}")
        nafnet_sd = torch.load(nafnet_ckpt, map_location="cpu")
        model_with_lora.NAFNet.load_state_dict(nafnet_sd, strict=True)
    return model_with_lora


def run_inference(model, model_wrap_cfg, null_token, input_image_pil, customize_text, args, seed):
    preprocess = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_image = preprocess(input_image_pil).unsqueeze(0).to(torch.float32).cuda()

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        customize_prompt_tokens = model.model_assessment.tokenizer(
            customize_text,
            truncation=True, max_length=77,
            return_length=False, return_overflowing_tokens=False,
            padding="max_length", return_tensors="pt"
        ).to(model.device)
        tokens = customize_prompt_tokens['input_ids'].cuda()
        customize_token = model.model_assessment.clipmodel.text_model(input_ids=tokens).last_hidden_state
        cond["c_crossattn"] = [customize_token]

        uncond = {
            "c_crossattn": [null_token],
            "c_concat":    [torch.zeros_like(cond["c_concat"][0])],
        }

        sigmas = K.external.CompVisDenoiser(model).get_sigmas(args.steps)
        extra_args = {
            "cond": cond, "uncond": uncond,
            "text_cfg_scale": args.cfg_text, "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)

        recon_stack = torch.cat((input_image, x), dim=1)
        result = model.NAFNet(recon_stack)

        correct_stable = adaptive_instance_normalization(x, input_image)
        correct_final  = adaptive_instance_normalization(result, input_image)

        def to_pil(t):
            t = torch.clamp((t + 1.0) / 2.0, min=0.0, max=1.0)
            t = 255.0 * rearrange(t, "1 c h w -> h w c")
            return Image.fromarray(t.type(torch.uint8).cpu().numpy())

        return {
            "result":           to_pil(x),
            "result_w_SCM":     to_pil(result),
            "colorcorrect":     to_pil(correct_stable),
            "SCM_colorcorrect": to_pil(correct_final),
        }


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution",  default=512,   type=int)
    parser.add_argument("--steps",       default=100,   type=int)
    parser.add_argument("--config",      default=CONFIG, type=str)
    parser.add_argument("--ckpt",        default=BASE_CKPT, type=str)
    parser.add_argument("--input-json",  default=INPUT_JSON, type=str)
    parser.add_argument("--output",      default=OUTPUT_DIR, type=str)
    parser.add_argument("--cfg-text",    default=1,     type=float)
    parser.add_argument("--cfg-image",   default=1,     type=float)
    parser.add_argument("--seed",        default=42,    type=int)
    args = parser.parse_args()

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    os.makedirs(args.output, exist_ok=True)

    # 读取 Qwen 分析结果
    with open(args.input_json, "r", encoding="utf-8") as f:
        qwen_results = json.load(f)
    print(f"Loaded {len(qwen_results)} Qwen analysis results")

    # 按退化类型分组
    grouped = defaultdict(list)
    for item in qwen_results:
        # 优先使用 qwen_response，否则使用 degradation_answer
        response = item.get("qwen_response") or item.get("degradation_answer") or ""
        deg_type = classify_degradation(response)
        item["degradation_type"] = deg_type
        item["_response"] = response
        grouped[deg_type].append(item)

    print(f"Degradation types found: {list(grouped.keys())}")
    for deg, items in grouped.items():
        print(f"  {deg}: {len(items)} images")

    # 加载 config
    config = OmegaConf.load(args.config)

    current_lora = None
    model = None

    for deg_type, items in grouped.items():
        lora_ckpt = LORA_MAP.get(deg_type)
        if lora_ckpt is None:
            print(f"WARNING: No LoRA for '{deg_type}', skipping {len(items)} samples")
            continue

        # Mix 退化类型特殊处理：不使用 LoRA，直接使用 base model
        is_mix = (deg_type == "Mix")
        
        if lora_ckpt != current_lora or (is_mix and current_lora != "__mix__"):
            print(f"\n=== Loading model for [{deg_type}]: {lora_ckpt} ===")
            if model is not None:
                del model
                torch.cuda.empty_cache()
            
            if is_mix:
                # Mix: 只加载 base model，不加载 LoRA
                model = load_base_model(config, MIX_BASE_CKPT)
                model.eval().cuda().to(torch.float32)
                current_lora = "__mix__"
            else:
                # 普通 LoRA 模式
                base_model = load_base_model(config, args.ckpt)
                model = attach_lora(base_model, lora_ckpt)
                model.eval().cuda().to(torch.float32)
                current_lora = lora_ckpt

            model_wrap = K.external.CompVisDenoiser(model)
            model_wrap_cfg = CFGDenoiser(model_wrap)

            null_tokens = model.model_assessment.tokenizer(
                '', truncation=True, max_length=77,
                return_length=False, return_overflowing_tokens=False,
                padding="max_length", return_tensors="pt"
            ).to('cuda')
            null_token = model.model_assessment.clipmodel.text_model(
                input_ids=null_tokens['input_ids']
            ).last_hidden_state

        for item in tqdm(items, desc=f"[{deg_type}]"):
            img_path = item["image_path"]
            img_name = os.path.splitext(item["image"])[0]
            response = item.get("_response", "")

            if not os.path.exists(img_path):
                # 尝试基于脚本所在目录的相对路径
                img_path_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path.lstrip("./"))
                if os.path.exists(img_path_abs):
                    img_path = img_path_abs
                else:
                    print(f"Warning: {item['image_path']} not found, skipping")
                    continue

            # 显示图片和退化信息
            print(f"\n{'='*60}")
            print(f"Image: {item['image']}")
            print(f"Degradation Type: {deg_type}")
            print(f"Degradation Answer: {response[:200]}...")

            # 处理 customize text
            customize = process_customize_text(response, deg_type)

            output_path = os.path.join(args.output, deg_type, img_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                outputs = run_inference(model, model_wrap_cfg, null_token, Image.open(img_path).convert("RGB"), customize, args, seed)
                outputs["result"].save(os.path.join(output_path + '_result.png'))
                outputs["result_w_SCM"].save(os.path.join(output_path + '_result_w_SCM.png'))
                outputs["colorcorrect"].save(os.path.join(output_path + '_result_colorcorrect.png'))
                outputs["SCM_colorcorrect"].save(os.path.join(output_path + '_result_w_SCM_colorcorrect.png'))
            except Exception as e:
                print(f"Error on {img_path}: {e}")
                continue

    print(f"\nAll done! Results saved to {args.output}")


if __name__ == "__main__":
    main()
