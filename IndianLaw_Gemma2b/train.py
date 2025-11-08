import os
import shutil
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from google.colab import drive

# --- 1. Install Dependencies ---
print(" STEP 1: Installing all requirements...")
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install "datasets<4.0.0"
!pip install --no-deps --upgrade "flash-attn>=2.6.3"
print("\n Installation complete. A kernel restart might be needed if this is the first run.")

# --- 2. Configuration ---
print("\n--> Configuring project parameters...")
MODEL_NAME = "unsloth/gemma-2-2b-it-bnb-4bit" 
DATASET_NAME = "Alok2304/Indian_Law_Final_Dataset" 
MAX_SEQ_LENGTH = 2048 
MAX_STEPS = 500

# --- Output Paths ---
HF_ADAPTER_DIR = "fine_tuned_IndianLaw_gemma2b_adapter" 
MERGED_DIR = "merged_IndianLaw_gemma2b_model_fp16" 
GGUF_OUTPUT_FILE = "IndianLaw_gemma2b_Q4_K_M.gguf" 
DRIVE_DEST_DIR = "/content/drive/MyDrive/My_IndianLaw_Gemma2b_Export"

# --- GGUF Naming Fix ---
DEFAULT_GGUF_NAME = "gemma-2-2b-it.Q4_K_M.gguf"
GGUF_TEMP_DIR = "gguf_conversion_output"

# --- 3. Load Model & Tokenizer ---
print(f"\n--> Loading base model '{MODEL_NAME}'...")
use_bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=dtype,
    load_in_4bit=True, use_gradient_checkpointing=True, token=None, trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("   - Set tokenizer pad_token to eos_token.")

# --- 4. Apply LoRA Adapters ---
print("\n--> Applying LoRA adapters to the model...")
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, 
    lora_dropout=0, # Set to 0 for Unsloth performance patch
    bias="none"
)

# --- 5. Load & Prepare Dataset ---
print(f"\n--> Loading and preparing the dataset: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, split="train")

def format_chat(example):
    text = example["text"]
    if "[INST]" in text and "[/INST]" in text:
        try:
            user = text.split("[INST]")[1].split("[/INST]")[0].strip()
            assistant = text.split("[/INST]")[1].strip()
            return {"text": f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>"}
        except IndexError:
             return {"text": f"<start_of_turn>user\nProvide information based on the context.<end_of_turn>\n<start_of_turn>model\n{text}<end_of_turn>"}
    return {"text": f"<start_of_turn>user\nProvide information based on the context.<end_of_turn>\n<start_of_turn>model\n{text}<end_of_turn>"}

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
print("   - Dataset formatting complete.")

# --- 6. Configure and Start Training ---
print("\n--> Configuring the trainer and starting fine-tuning...")
sft_config = SFTConfig(
    output_dir="training_output", per_device_train_batch_size=2, gradient_accumulation_steps=8,
    max_steps=MAX_STEPS, learning_rate=2e-4, logging_steps=20, save_strategy="no",
    report_to="none", dataset_text_field="text", 
    packing=False, # Disabled packing to fix sequence length errors
    fp16=not use_bf16, bf16=use_bf16,
)

trainer = SFTrainer(
    model=model, args=sft_config, train_dataset=dataset, tokenizer=tokenizer, max_seq_length=MAX_SEQ_LENGTH
)

trainer.train()
print("✅ Fine-tuning complete.")

# --- 7. Save All Artifacts ---
print(f"\n--> Saving artifacts, merging, and converting to GGUF...")
print(f"   - Saving LoRA adapter to '{HF_ADAPTER_DIR}'...")
model.save_pretrained(HF_ADAPTER_DIR)
tokenizer.save_pretrained(HF_ADAPTER_DIR)

print(f"   - Merging adapter into a new model at '{MERGED_DIR}'...")
try:
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print("   - ✅ Merge successful.")
except Exception as e:
    print(f"   - ❌ ERROR: Could not merge the model. Error: {e}")

print(f"   - Converting model to GGUF...")
try:
    os.makedirs(GGUF_TEMP_DIR, exist_ok=True)
    model.save_pretrained_gguf(GGUF_TEMP_DIR, tokenizer, quantization_method="q4_k_m")
    print("   - ✅ GGUF conversion successful.")
except Exception as e:
    print(f"   - ❌ ERROR: GGUF export failed. Error: {e}")

# --- 8. Rename GGUF File ---
print("   - Finding and renaming GGUF file...")
try:
    # Unsloth saves to the /content directory, not the temp folder
    original_gguf_path = os.path.join("/content", DEFAULT_GGUF_NAME) 
    final_gguf_path = os.path.join("/content", GGUF_OUTPUT_FILE)

    if os.path.exists(original_gguf_path):
        shutil.move(original_gguf_path, final_gguf_path)
        print(f"   - ✅ Found and renamed GGUF to '{final_gguf_path}'")
    else:
        print(f"   - ⚠️ WARN: Could not find default GGUF file at '{original_gguf_path}'")
except Exception as e:
    print(f"   - ❌ ERROR: Failed to rename GGUF file. Error: {e}")

# --- 9. Mount Drive & Copy All Artifacts ---
print("\n--> Saving all final files to Google Drive...")
try:
    drive.mount('/content/drive', force_remount=True)
    os.makedirs(DRIVE_DEST_DIR, exist_ok=True)
    print(f"   - Created/found destination folder: {DRIVE_DEST_DIR}")

    artifacts_to_copy = [HF_ADAPTER_DIR, MERGED_DIR, f"/content/{GGUF_OUTPUT_FILE}"]
    for item_path in artifacts_to_copy:
        if os.path.exists(item_path):
            dest_path = os.path.join(DRIVE_DEST_DIR, os.path.basename(item_path))
            print(f"   - Copying '{item_path}' to Drive...")
            if os.path.isdir(item_path):
                if os.path.exists(dest_path): shutil.rmtree(dest_path)
                shutil.copytree(item_path, dest_path)
            else:
                shutil.copy2(item_path, dest_path)
        else:
            print(f"   - ⚠️ WARN: Artifact not found, cannot copy: {item_path}")
    
    print("\n SUCCESS! All artifacts have been copied to your Google Drive.")

except Exception as e:
    print(f"❌ ERROR: Failed to mount or copy to Google Drive. Error: {e}")

print("\n=== ALL STEPS FINISHED ===")
