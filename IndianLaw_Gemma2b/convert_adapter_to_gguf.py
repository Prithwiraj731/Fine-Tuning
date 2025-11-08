import os
import shutil
import torch
from unsloth import FastLanguageModel
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
MAX_SEQ_LENGTH = 2048

# --- Paths for loading and saving ---
DRIVE_EXPORT_DIR = "/content/drive/MyDrive/My_IndianLaw_Gemma2b_Export"
ADAPTER_FROM_DRIVE = os.path.join(DRIVE_EXPORT_DIR, "fine_tuned_IndianLaw_gemma2b_adapter")

# --- Paths for GGUF output ---
DEFAULT_GGUF_NAME = "gemma-2-2b-it.Q4_K_M.gguf"
FINAL_GGUF_NAME = "IndianLaw_gemma2b_Q4_K_M.gguf"
GGUF_TEMP_DIR = "gguf_conversion_output"
GDRIVE_GGUF_DEST = '/content/drive/MyDrive/gguf_models' # Final GGUF destination

# --- 3. Mount Drive ---
print("\n--> Mounting Google Drive...")
try:
    drive.mount('/content/drive', force_remount=True)
    print("   - ✅ Drive mounted.")
except Exception as e:
    print(f"   - ❌ ERROR: Could not mount Drive. Stopping script. Error: {e}")
    exit()

# --- 4. Load Base Model ---
print(f"\n--> Loading base model '{MODEL_NAME}'...")
use_bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=dtype,
    load_in_4bit=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("   - Set tokenizer pad_token to eos_token.")

# --- 5. Load Your Fine-Tuned Adapter ---
print(f"\n--> Loading fine-tuned adapter from: {ADAPTER_FROM_DRIVE}...")
if os.path.exists(ADAPTER_FROM_DRIVE):
    print("   - Step 1: Applying new LoRA adapters to make model a PeftModel...")
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0, bias="none"
    )
    print("   - Step 2: Loading saved adapter weights...")
    model.load_adapter(ADAPTER_FROM_DRIVE, adapter_name="default")
    print("   - ✅ Successfully loaded adapter.")
else:
    print(f"   - ❌ ERROR: Adapter not found at '{ADAPTER_FROM_DRIVE}'. Cannot proceed.")
    exit()

# --- 6. Convert to GGUF ---
print(f"\n--> Converting model to GGUF (q4_k_m)...")
try:
    os.makedirs(GGUF_TEMP_DIR, exist_ok=True)
    model.save_pretrained_gguf(GGUF_TEMP_DIR, tokenizer, quantization_method="q4_k_m")
    print("   - ✅ GGUF conversion successful.")
except Exception as e:
    print(f"   - ❌ ERROR: GGUF export failed. Error: {e}")

# --- 7. Find, Rename, and Move the GGUF File ---
print("\n--> Finding and renaming final GGUF file...")
try:
    # Unsloth saves the file to the /content directory
    original_gguf_path = os.path.join("/content", DEFAULT_GGUF_NAME)
    final_gguf_path = os.path.join("/content", FINAL_GGUF_NAME) 

    if os.path.exists(original_gguf_path):
        shutil.move(original_gguf_path, final_gguf_path)
        print(f"   - ✅ Found and moved GGUF to '{final_gguf_path}'")
    else:
        print(f"   - ⚠️ WARN: Could not find default GGUF file at '{original_gguf_path}'")
except Exception as e:
    print(f"   - ❌ ERROR: Failed to rename GGUF file. Error: {e}")

# --- 8. Copy Final GGUF to Google Drive ---
print(f"\n--> Copying '{FINAL_GGUF_NAME}' to Google Drive...")
os.makedirs(GDRIVE_GGUF_DEST, exist_ok=True)
src_path = f'/content/{FINAL_GGUF_NAME}'

if os.path.exists(src_path):
    dest_path = os.path.join(GDRIVE_GGUF_DEST, FINAL_GGUF_NAME)
    print(f"   - Copying {FINAL_GGUF_NAME}...")
    shutil.copy2(src_path, dest_path)
    file_size = os.path.getsize(dest_path) / (1024**3) # Convert to GB
    print(f"   - ✅ {FINAL_GGUF_NAME} copied successfully ({file_size:.2f} GB)")
else:
    print(f"   - ❌ ERROR: Final GGUF file not found at {src_path}")

print(f"\n[INFO] All steps finished. Find your file at: {GDRIVE_GGUF_DEST}")
