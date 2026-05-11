# =========================================================
# Qwen2.5-VL-7B LoRA Finetune for DepictQA
# =========================================================

import os
# 使用四块卡训练
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_DATASETS_CACHE"] = "./CoTIR/cache/datasets_cache"
os.environ["TMPDIR"] = "./CoTIR/cache/datasets_tmp"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import time
import logging
import json
from PIL import Image
import torch
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

# =======================
# 1. LOG
# =======================
LOG_DIR = "./CoTIR/experience/depictqa_lora"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
        ),
        logging.StreamHandler(),
    ],
)
logging.info("===== Start DepictQA LoRA Training =====")

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(
                " | ".join(
                    [f"step={state.global_step}"]
                    + [
                        f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in logs.items()
                    ]
                )
            )

# =======================
# 2. PATH CONFIG
# =======================
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_DIR = "./CoTIR/cache"
# DepictQA JSON文件路径
DATA_JSON = "./DepictQA_Single_Degradation_Q1_Answer_Only_final.json"
# 图像根目录（JSON中target_image是相对路径）
IMAGE_BASE_DIR = "./depictqa_package"
# 输出目录
OUTPUT_DIR = "./checkpoint_depictqa_lora"
MAX_LENGTH = 8192

# =======================
# 3. LOAD DATASET
# =======================
logging.info("Loading DepictQA dataset...")

with open(DATA_JSON, "r") as f:
    raw_data = json.load(f)

logging.info(f"Loaded {len(raw_data)} samples from {DATA_JSON}")

# 转换为HuggingFace Dataset格式
def convert_to_messages(item):
    """将JSON转换为messages格式"""
    # 构建图像路径
    image_path = os.path.join(IMAGE_BASE_DIR, item["target_image"])
    
    # 构建对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": item["query"]}
            ]
        },
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": item["answer"]}
            ]
        }
    ]
    return {"messages": messages, "image_path": image_path}

# 过滤掉不存在的图像
valid_data = []
for item in raw_data:
    image_path = os.path.join(IMAGE_BASE_DIR, item["target_image"])
    if os.path.exists(image_path):
        valid_data.append(convert_to_messages(item))

logging.info(f"Valid samples with existing images: {len(valid_data)}/{len(raw_data)}")

# 创建Dataset
dataset = Dataset.from_list(valid_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# =======================
# 4. LOAD MODEL (开启 Flash Attention 2 提升效率)
# =======================
logging.info("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    local_files_only=True
)
model.config.use_cache = False
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True)

# =======================
# 5. PROCESS FUNCTION
# =======================
import numpy as np
import os

CACHE_DIR = "/public/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def process_func(example, idx):
    messages = example["messages"]
    
    # 打开图像
    image = Image.open(example["image_path"]).convert("RGB")
    
    # 获取 query 和 answer
    query = messages[0]["content"][1]["text"]
    answer = messages[1]["content"][0]["text"]
    
    # 构建用户消息 - 先声明图像占位符，再加文本
    user_content = [
        {"type": "image"},
        {"type": "text", "text": query}
    ]
    
    # 完整的对话
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    
    # 生成 Chat 文本（包含 generation prompt）
    text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
    
    # 处理输入 - 关键：使用 list 格式
    model_inputs = processor(
        text=[text],
        images=[[image]],  # 外层list对应batch，内层list对应多张图像
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    
    input_ids = model_inputs["input_ids"][0].tolist()
    
    # Label 处理：只对答案计算 Loss，加上 EOS
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    
    final_input_ids = (input_ids + answer_ids)[:MAX_LENGTH]
    final_labels = ([-100] * len(input_ids) + answer_ids)[:MAX_LENGTH]
    
    # 保存 pixel_values 到缓存文件，避免 Arrow 溢出
    if "pixel_values" in model_inputs:
        pv_path = os.path.join(CACHE_DIR, f"pv_{idx}.npy")
        np.save(pv_path, model_inputs["pixel_values"].numpy())
        
        grid_thw_path = os.path.join(CACHE_DIR, f"grid_{idx}.npy")
        np.save(grid_thw_path, model_inputs["image_grid_thw"].numpy())
        
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "attention_mask": [1] * len(final_input_ids),
            "pixel_values_path": pv_path,
            "image_grid_thw_path": grid_thw_path,
        }
    
    return {
        "input_ids": final_input_ids,
        "labels": final_labels,
        "attention_mask": [1] * len(final_input_ids),
    }

def process_func_wrapper(example, idx):
    return process_func(example, idx)

logging.info("Processing datasets...")
# 缓存路径
train_cache = os.path.join(CACHE_DIR, "train_ds.arrow")
eval_cache = os.path.join(CACHE_DIR, "eval_ds.arrow")

# 禁用并行处理，避免 Arrow 溢出（pixel_values 太大）
train_ds = train_ds.map(process_func_wrapper, remove_columns=train_ds.column_names, with_indices=True, cache_file_name=train_cache)
eval_ds = eval_ds.map(process_func_wrapper, remove_columns=eval_ds.column_names, with_indices=True, cache_file_name=eval_cache)

# =======================
# 6. LoRA CONFIG
# =======================
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =======================
# 7. DATA COLLATOR
# =======================
import numpy as np

def vl_data_collator(features):
    batch = {}
    
    batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f["input_ids"]) for f in features],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    batch["labels"] = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f["labels"]) for f in features],
        batch_first=True,
        padding_value=-100,
    )
    batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f["attention_mask"]) for f in features],
        batch_first=True,
        padding_value=0,
    )
    
    # 从缓存文件加载 pixel_values 和 image_grid_thw（使用 mmap 加速）
    if "pixel_values_path" in features[0]:
        pv_list = []
        grid_list = []
        for f in features:
            pv_arr = np.load(f["pixel_values_path"], mmap_mode='r')
            grid_arr = np.load(f["image_grid_thw_path"], mmap_mode='r')
            pv_list.append(np.array(pv_arr))  # 转为普通数组
            grid_list.append(np.array(grid_arr))
        
        batch["pixel_values"] = torch.from_numpy(np.concatenate(pv_list, axis=0))
        batch["image_grid_thw"] = torch.from_numpy(np.concatenate(grid_list, axis=0))
    
    return batch

# =======================
# 8. TRAINING ARGS (四卡训练配置)
# =======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,      # 增大batch_size
    gradient_accumulation_steps=4,      # 调整累积步数
    num_train_epochs=2,                 # 训练2个epoch
    learning_rate=5e-5,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    dataloader_num_workers=8,
    report_to="tensorboard",
    # 多卡分布式训练
    local_rank=-1,
    ddp_find_unused_parameters=False,
)

# =======================
# 9. TRAINER
# =======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=vl_data_collator,
    callbacks=[
        EarlyStoppingCallback(5),
        LoggingCallback(),
    ],
)

logging.info("Starting training...")
logging.info(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")
trainer.train()
logging.info("Training finished successfully!")
