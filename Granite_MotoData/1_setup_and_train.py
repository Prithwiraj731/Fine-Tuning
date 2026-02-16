import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from google.colab import drive
from transformers import TrainingArguments, AutoConfig
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

# ==========================================
# ⚙️ USER CONFIGURATION (CHANGE THESE)
# ==========================================
# 1. The name of your uploaded JSONL file
DATASET_FILE = "Your_Dataset.jsonl"  # <--- REPLACE THIS WITH YOUR FILE NAME

# 2. Where to save the model in Google Drive
OUTPUT_DIR = "/content/drive/MyDrive/My_Granite_FineTune"
# ==========================================

# 1. Mount Google Drive
drive.mount('/content/drive')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Model & Disable Dropout (Fixes Granite bugs)
max_seq_length = 4096
config = AutoConfig.from_pretrained("ibm-granite/granite-3.1-2b-instruct")
config.attention_dropout = 0.0
config.hidden_dropout = 0.0

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ibm-granite/granite-3.1-2b-instruct",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    config = config,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Load & Format Dataset
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"❌ DATASET MISSING: Please upload '{DATASET_FILE}' to the Colab file list!")

data = []
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            # Robust list handling
            for key in ["answer", "output", "response"]:
                if key in item and isinstance(item[key], list):
                    item[key] = " ".join(str(x) for x in item[key])
            data.append(item)
        except json.JSONDecodeError:
            continue

dataset = Dataset.from_list(data)

# Apply Chat Template
def formatting_prompts_func(examples):
    texts = []
    for i in range(len(examples[list(examples.keys())[0]])):
        # Handles both 'question'/'input' and 'answer'/'output' keys
        q = examples.get("question", examples.get("input", [""]))[i]
        a = examples.get("answer", examples.get("output", [""]))[i]
        
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(a)}],
            tokenize = False, add_generation_prompt = False
        )
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Start Training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,  # Increase this for longer training like 500 OR 600
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

print("🚀 Starting Training...")
trainer.train()

# 6. Save Adapters
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")
print(f"✅ Training Complete. Adapters saved to {OUTPUT_DIR}")
