# A Collection of Fine-Tuned Language Models 🚀

Welcome to my portfolio of fine-tuning projects! This repository serves as a central hub for the code and notebooks I've used to specialize various open-source language models for specific tasks.

Each project is contained within its own folder in this repository. The finished, ready-to-use models (in GGUF format) are hosted on my Hugging Face profile for easy access.

---

## 📂 My Fine-Tuned Models

This table provides a quick overview of each project, with direct links to the code and the final model.

| Project | Base Model | Code Location (GitHub) | Final Model (Hugging Face) |
| :--- | :--- | :--- | :--- |
| **Medical Dialogue Summarizer** 🧑‍⚕️ | `microsoft/phi-2` | [./Medical_Phi2/](./Medical_Phi2/) |  [**Prithwiraj731/Medical-Phi2-GGUF**](https://huggingface.co/Prithwiraj731/Medical-Phi2-GGUF) |
| **Motorcycle & Law Assistant** 🏍️ | `microsoft/phi-2` | [./MotoData_Phi2/](./MotoData_Phi2/) |  [**Prithwiraj731/MotoData-Phi2-GGUF**](https://huggingface.co/Prithwiraj731/MotoData-Phi2-GGUF) |
| *(Soon)* | *(Base Model)* | *(#)* | *(#)* |

---

## 🛠️ How This Repository is Structured

This repository is organized to be simple and reproducible:

1.  **Code is Here (The Recipe 📜):** Each folder (e.g., `Medical_Phi2`) contains the complete Google Colab notebook (`.ipynb` file) with the exact code used for the fine-tuning, merging, and conversion process. This allows you to see my methodology and reproduce the results.

2.  **Models are on Hugging Face (The Finished Cake 🎂):** The final, ready-to-use GGUF files are hosted on the Hugging Face Hub. This is the best practice, as it keeps this GitHub repository lightweight and uses the best platform for model distribution.

---

## 💻 General Guide: How to Use the GGUF Models

All GGUF models from my Hugging Face repositories can be run on your local computer using fantastic free tools like LM Studio or Ollama.

### Using with LM Studio (Easiest Method)

1.  Download and install [**LM Studio**](https://lmstudio.ai/).
2.  Open the app and use the search bar (🔍 icon) to find one of my models (e.g., `Prithwiraj731/Medical-Phi2-GGUF`).
3.  From the file list on the right, click **Download** next to the GGUF file.
4.  Go to the chat tab (💬 icon), select the model you downloaded at the top, and start your conversation!



### Using with Ollama (Advanced Method)

1.  Download and install [**Ollama**](https://ollama.com/).
2.  Download the desired GGUF file from one of my Hugging Face repositories.
3.  Create a file named `Modelfile` (no extension) in the same directory as your downloaded GGUF file. Paste the following into it, **replacing the placeholder names**:

    ```
    # Replace the filename with the one you downloaded
    FROM ./your-model-name.gguf

    # Use the prompt template from the model's Hugging Face page
    TEMPLATE "<start_of_turn>user\n{{ .Prompt }}<end_of_turn>\n<start_of_turn>model\n"
    ```
4.  Open your terminal and create the model by running:
    ```bash
    # Replace 'YourModelName' with a name you choose
    ollama create YourModelName -f ./Modelfile
    ```
5.  You can now chat with your model anytime by running:
    ```bash
    ollama run YourModelName
    ```
---

*Projects fine-tuned by Prithwiraj731.*
