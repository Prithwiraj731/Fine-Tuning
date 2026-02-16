from huggingface_hub import HfApi, login
from google.colab import drive
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
HF_TOKEN = "hf_..."  # <--- PASTE YOUR WRITE TOKEN HERE
NEW_REPO_ID = "YourUsername/Granite-3.1-2b-MyDataset" # <--- NAME YOUR REPO

# The folder containing your GGUFs and adapters
MODEL_FOLDER = "/content/drive/MyDrive/My_Granite_FineTune"
# ==========================================

drive.mount('/content/drive')
login(token=HF_TOKEN)
api = HfApi()

print(f"🚀 Uploading {MODEL_FOLDER} to {NEW_REPO_ID}...")

if not os.path.exists(MODEL_FOLDER):
    print("❌ Error: Folder not found.")
else:
    try:
        api.create_repo(repo_id=NEW_REPO_ID, private=False, exist_ok=True)
        
        api.upload_folder(
            folder_path=MODEL_FOLDER,
            repo_id=NEW_REPO_ID,
            repo_type="model",
            commit_message="Upload Fine-Tuned Granite Model + GGUFs"
        )
        print("🎉 Success! Your model is live.")
        print(f"🔗 Link: https://huggingface.co/{NEW_REPO_ID}")
    except Exception as e:
        print(f"❌ Upload Error: {e}")
