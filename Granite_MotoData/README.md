# 🧠 Generic Granite 3.1 2B Fine-Tuner

This repository contains a modular, reusable workflow for fine-tuning the **IBM Granite 3.1 2B Instruct** model on *any* custom dataset using **Unsloth**.

It is designed to be plug-and-play: just drop in your dataset, run the scripts, and get a fully quantized GGUF model ready for local use.

## 🚀 Features
- **Universal Dataset Support:** Works with any `.jsonl` file containing `question/answer` or `input/output` pairs.
- **Bug-Free Granite Config:** Automatically handles the "dropout" and "flex_attention" errors common with Granite models.
- **Auto-Quantization:** Converts the trained model to **GGUF (Q4_K_M & FP16)** for use in LM Studio or Ollama.
- **Drive Backup:** Safely saves progress to Google Drive.

## 📂 Project Structure
* `1_setup_and_train.py`: Installs dependencies, loads the model, and runs the training loop.
* `2_convert_to_gguf.py`: Merges the LoRA adapters and converts weights to GGUF format.
* `3_push_to_huggingface.py`: Uploads the final models to your Hugging Face profile.

## 🛠️ How to Use

### 1. Prepare Your Dataset
Create a file named `dataset.jsonl`. It should look like this:
json
{"question": "What is the best engine oil for a classic bike?", "answer": "For classic bikes, 20W-50 mineral oil is often recommended..."}
{"question": "How do I fix a flat tire?", "answer": "First, locate the puncture..."}

2. Run Training (1_setup_and_train.py)

    Open the script.

    Change DATASET_FILE = "dataset.jsonl" to match your filename.

    Run the script. It will fine-tune the model and save adapters to your Drive.

3. Convert (2_convert_to_gguf.py)

    Run this script to generate granite-2b-q4_k_m.gguf.

    This file is compatible with LM Studio, Ollama, and llama.cpp.

4. Upload (3_push_to_huggingface.py)

    Add your Hugging Face token.

    Run the script to share your model with the world.

❤️ Credits

Powered by Unsloth AI and IBM Granite.
