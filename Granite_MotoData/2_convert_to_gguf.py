import os
import shutil
from unsloth import FastLanguageModel
from google.colab import drive

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Where your adapters are saved (Same as OUTPUT_DIR in File 2)
ADAPTER_PATH = "/content/drive/MyDrive/My_Granite_FineTune/lora_adapters"

# Where to put the final GGUF files
FINAL_DESTINATION = "/content/drive/MyDrive/My_Granite_FineTune"
# ==========================================

drive.mount('/content/drive')

print("🔄 Loading Model for Conversion...")
model, tokenizer = FastLanguageModel.from_pretrained(
    ADAPTER_PATH,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

# Create a local temporary folder (Faster & safer than Drive)
local_temp = "/content/temp_conversion"
if os.path.exists(local_temp): shutil.rmtree(local_temp)
os.makedirs(local_temp, exist_ok=True)

# Save tokenizer locally for llama.cpp
tokenizer.save_pretrained(local_temp)

print("⏳ Converting to GGUF (this takes a few minutes)...")
try:
    # 1. Convert to FP16
    print("   -> Generating FP16...")
    model.save_pretrained_gguf(local_temp, tokenizer, quantization_method = "f16")
    
    # 2. Convert to Q4_K_M (Recommended)
    print("   -> Generating Q4_K_M...")
    model.save_pretrained_gguf(local_temp, tokenizer, quantization_method = "q4_k_m")
    
    print("✅ Conversion Done. Moving files to Drive...")
    
    # Move files
    for file in os.listdir(local_temp):
        if file.endswith(".gguf"):
            shutil.copy2(os.path.join(local_temp, file), os.path.join(FINAL_DESTINATION, file))
            print(f"   -> Saved: {file}")

    print(f"🎉 All GGUF files are safe in: {FINAL_DESTINATION}")

except Exception as e:
    print(f"❌ Conversion Failed: {e}")
