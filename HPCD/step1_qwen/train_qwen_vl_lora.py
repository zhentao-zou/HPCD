# =========================================================
# Qwen2.5-VL-7B LoRA Finetune for DepictQA
# =========================================================

import os
# 设置单卡训练（如果要使用多卡，通过 torchrun 启动脚本设置）
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 注释掉，让 torchrun 自动管理
os.environ["HF_DATASETS_CACHE"] = "./datasets_cache"
os.environ["TMPDIR"] = "./cache/datasets_tmp"

import json
import logging
import time
import numpy as np
from PIL import Image
from typing import Dict, List, Any
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

# =======================
# 1. LOG 配置
# =======================
LOG_DIR = "/public/zzt/zzt/CoTIR/experience/depictqa_lora"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# =======================
# 2. 路径配置
# =======================
MODEL_PATH = "./model/Qwen2.5-VL-7B-Instruct"
DATA_JSON = "./depictqa_package/DepictQA_Single_Degradation_Q1_Answer_Only_final.json"
IMAGE_BASE_DIR = "./AutoDIR-main/depictqa_package"
OUTPUT_DIR = "./checkpoint_depictqa_lora"
MAX_LENGTH = 2048*2  # 减小最大长度节省显存

# 缓存目录
# =======================
# 3. 加载模型
# =======================
logger.info("Loading model and processor...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# =======================
# 4. LoRA 配置
# =======================
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =======================
# 5. DataCollator
# =======================
class VLDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        try:
            self.image_token_id = tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index("<|image_pad|>")
            ] if "<|image_pad|>" in tokenizer.additional_special_tokens else None
        except:
            self.image_token_id = None
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 找出最大长度并 pad
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Pad input_ids
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # labels 处理：pad token 和 image token 设为 -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
        
        # 收集 pixel_values 和 image_grid_thw
        pixel_values_list = []
        grid_list = []
        for f in features:
            pv = f["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv)
            pixel_values_list.append(pv)
            
            gw = f["image_grid_thw"]
            if not isinstance(gw, torch.Tensor):
                gw = torch.tensor(gw)
            grid_list.append(gw)
        
        # 合并 pixel_values
        max_pv_len = max(pv.shape[0] for pv in pixel_values_list)
        padded_pv = []
        for pv in pixel_values_list:
            if pv.shape[0] < max_pv_len:
                pad = torch.zeros((max_pv_len - pv.shape[0], pv.shape[1]), dtype=pv.dtype)
                pv = torch.cat([pv, pad], dim=0)
            padded_pv.append(pv)
        pixel_values = torch.stack(padded_pv, dim=0)
        
        # 合并 image_grid_thw
        image_grid_thw = torch.stack(grid_list, dim=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

# =======================
# 6. 数据加载
# =======================
logger.info("Loading dataset...")

with open(DATA_JSON, "r") as f:
    raw_data = json.load(f)

logger.info(f"Total samples: {len(raw_data)}")

# 划分数据集
train_data, eval_data = train_test_split(raw_data, test_size=0.1, random_state=42)

logger.info(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

# =======================
# 7. 自定义 Dataset（实时加载）
# =======================
class VLRealtimeDataset(Dataset):
    def __init__(self, data_list, tokenizer, processor):
        self.data = data_list
        self.tokenizer = tokenizer
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image_path = os.path.join(IMAGE_BASE_DIR, example["target_image"].lstrip("./"))
        
        if not os.path.exists(image_path):
            return None

        # 1. 构造消息：只包含 User 部分，用于生成 Prompt 模板
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "resized_height": 256, "resized_width": 256},
                    {"type": "text", "text": example["query"]}
                ]
            }
        ]
        
        # 2. 生成 Prompt 文本（包含 <|im_start|>assistant\n 提示符）
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 3. 处理视觉信息和 Prompt Token
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False, # 这里设为 False，由 Collator 统一处理
            return_tensors="pt",
        )
        
        prompt_ids = inputs["input_ids"].squeeze(0).tolist()
        
        # 4. 处理答案：手动加上结束符 <|im_end|>
        # Qwen2.5 使用 <|im_end|> 作为对话结束标志
        answer_text = example["answer"] + self.tokenizer.eos_token 
        response = self.tokenizer(answer_text, add_special_tokens=False)
        answer_ids = response["input_ids"]
        
        # 5. 拼接：Prompt 部分 + Answer 部分
        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        
        # 6. 构造 Labels：Prompt 部分设为 -100，Answer 部分保留
        labels = [-100] * len(prompt_ids) + answer_ids
        
        # 7. 长度截断
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
        }

# =======================
# 7. 创建 Dataset
# =======================
logger.info("Creating datasets...")

train_dataset = VLRealtimeDataset(train_data, tokenizer, processor)
eval_dataset = VLRealtimeDataset(eval_data, tokenizer, processor)

logger.info(f"Training samples: {len(train_dataset)}")
logger.info(f"Eval samples: {len(eval_dataset)}")

# =======================
# 8. 训练参数
# =======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,  # 减小 batch size
    gradient_accumulation_steps=1,  # 增加累积步数
    num_train_epochs=4,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=4000,
    save_steps=1000,
    save_total_limit=10,
    load_best_model_at_end=False,
    dataloader_num_workers=4,
    report_to="tensorboard",
    # local_rank 和 ddp_find_unused_parameters 不需要手动设置
    # 使用 torchrun 启动时会自动设置
    remove_unused_columns=False,  # 保留所有列，因为我们的 DataCollator 需要 pixel_values_path
)

# =======================
# 9. Trainer
# =======================
data_collator = VLDataCollator(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# =======================
# 10. 开始训练
# =======================
logger.info("=" * 50)
logger.info(f"Starting training...")
logger.info(f"Train samples: {len(train_dataset)}")
logger.info(f"Eval samples: {len(eval_dataset)}")
logger.info("=" * 50)

trainer.train()

logger.info("Training finished!")
