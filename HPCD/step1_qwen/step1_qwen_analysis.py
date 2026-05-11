"""
Step 1: 使用 Qwen2.5-VL + LoRA 分析 LR 图片的退化信息，保存为 JSON
"""
import os
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

# 配置
MODEL_PATH = "/public/zzt/zzt/CoTIR/merged_qwen25vl_v1"  # 本地模型路径 

PEFT_MODEL_PATH = "./checkpoints/qwen_lora/"  # LoRA 权重路径
INPUT_DIR = "./inputs/LR/"                       # 输入图片目录
OUTPUT_JSON = "./outputs/qwen_analysis_results.json"  # 输出 JSON

# 查询问题（与微调时一致）
QUERY_TEXT = 'What distortions are most prominent in the evaluated image, and infer the main topographical and environmental features of the image if the distortions are be eliminated, please answer these questions within 80 word?'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def predict(messages, model, processor):
    """推理函数"""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[messages[0]['content'][0]['image']],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    input_ids_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, input_ids_len:]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output_text


def resize_image(image, max_side=512):
    """调整图片大小，保持宽高比"""
    max_dim = max(image.size)
    if max_dim > max_side:
        ratio = max_side / max_dim
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image


print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading fine-tuned model...")
try:
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
    peft_model.eval()
    print("Fine-tuned model loaded successfully!")
except Exception as e:
    print(f"Error loading LoRA model: {e}")
    exit()

# 获取图片列表
image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {len(image_files)} images")

results = []

for img_file in tqdm(image_files, desc="Analyzing"):
    img_path = os.path.join(INPUT_DIR, img_file)

    try:
        image = Image.open(img_path).convert("RGB")
        image = resize_image(image)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QUERY_TEXT},
            ]}
        ]

        response = predict(messages, peft_model, processor)

        result = {
            "image": img_file,
            "image_path": img_path,
            "qwen_response": response,
        }
        results.append(result)

        print(f"\n{'='*60}")
        print(f"Image: {img_file}")
        print(f"Response: {response}")

    except FileNotFoundError:
        print(f"Warning: Image file not found at {img_path}. Skipping.")
        continue
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "image": img_file,
            "image_path": img_path,
            "qwen_response": f"Error: {str(e)}",
        })

# 保存结果
os.makedirs(os.path.dirname(OUTPUT_JSON) if os.path.dirname(OUTPUT_JSON) else ".", exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to {OUTPUT_JSON}")
print(f"Total images processed: {len(results)}")
