# LLaMA-LoRA-FineTuning

A practical guide to fine-tuning Meta’s LLaMA language model using LoRA (Low-Rank Adaptation) for efficient, low-resource training. This project walks through the setup, training, and inference process using Hugging Face's `transformers`, `peft`, and `accelerate` libraries.

---

## Overview

Fine-tuning large language models like LLaMA typically requires vast computational resources. LoRA offers a lightweight solution by freezing most of the model weights and only fine-tuning small rank matrices, drastically reducing the compute burden.

This repository contains:
- Two well-documented Jupyter Notebooks:
  - `llama_lora_finetuning.ipynb` – fine-tuning LLaMA using LoRA.
  - `llama_lora_inference.ipynb` – running inference on the fine-tuned model.
- Easy-to-follow steps to reproduce results on your own dataset.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/LLaMA-LoRA-FineTuning.git
cd LLaMA-LoRA-FineTuning
pip install -r requirements.txt
```
### Run notebooks
You can open and run the notebooks directly via Jupyter or Google Colab.

### Requirements
- All required dependencies to run this project successfully has been given in the `requirements.txt`.
- Install them using instructions given in installation phase for successful running of the project.

---

## Project Structure

```bash
LLaMA-LoRA-FineTuning/
│
├── llama_lora_finetuning.ipynb    # Notebook for training with LoRA
├── llama_lora_inference.ipynb     # Notebook for inference
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and documentation
├── License                        # MIT open-source licensing
```
---

## Example Use Case

```python
# Sample inference after LoRA fine-tuning
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path_to_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("path_to_tokenizer")

prompt = "Once upon a time in AI land,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
---

## References
- [Fine-Tuning LLaMA with LoRA – ClearIntelligence Blog](https://clearintelligence.substack.com/p/fine-tuning-llama-llm-with-lora-a)
- [LoRA Paper – Microsoft Research](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)

---

## License
This project is licensed under [MIT license](https://github.com/veydantkatyal/Llama-LoRA-FineTuning/blob/main/LICENSE)
