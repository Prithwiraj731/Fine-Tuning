# Fine-Tuned Gemma 2B for Indian Law

This repository contains the scripts used to fine-tune the `unsloth/gemma-2-2b-it-bnb-4bit` model on the `Alok2304/Indian_Law_Final_Dataset`.

## 🚀 Model Repo on Hugging Face

All model files (GGUF, Merged, and Adapter) are hosted on Hugging Face.

**Find the models here:** [**https://huggingface.co/Prithwiraj731/IndianLaw-gemma2b-full**](https://huggingface.co/Prithwiraj731/IndianLaw-gemma2b-full)

---

## 💻 How to Use These Scripts

### 1. `train.py`
This script will run a full training process from scratch. It downloads the base model and dataset, trains for 500 steps, and saves the final adapter, merged model, and GGUF file to your Google Drive.

### 2. `convert_adapter_to_gguf.py`
Use this script if you have already trained and have the adapter in your Google Drive. It loads the adapter, converts it to GGUF (q4_k_m), and saves it to a `gguf_models` folder on your Drive.

### Installation

To run these scripts, first install the requirements:
```bash
pip install -r requirements.txt
