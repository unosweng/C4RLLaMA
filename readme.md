# Code Comment Inconsistency Detection and Rectification Using a Large Language Model

This repository contains the supplementary material for the paper:

> **"Code Comment Inconsistency Detection and Rectification Using a Large Language Model"**

## 📂 Repository Structure

- **`Data.7z`** – Compressed dataset used in the study.  
- **`templates/`** – LLaMA prompt templates.  
- **`utils/BalanceTrainer.py`** – Custom loss function used in training.  
- **`utils/prompter.py`** – Prompt formatting utility.  
- **`train.py`** – Main training script.  
- **`test.py`** – Model evaluation script.  
- **`run-train.sh`** – Single-node training script.  
- **`run-ddp.sh`** – Multi-GPU distributed training script (uses `torchrun`).  
- **`LoraCodeLlama_7B/`** – Output directory for LoRA fine-tuned weights (created after training).

---

## ⚙️ Setup

### 1. Install dependencies
We recommend using **conda**:
```bash
conda create -n c4rllama python=3.10
conda activate c4rllama
pip install -r requirements.pip.txt
````

### 2. Extract the dataset

```bash
7z x Data.7z -oData
```

---

## 🚀 Training

### **Option 1 – Single GPU**

```bash
bash run-train.sh
```

Or directly:

```bash
python -u train.py \
  --base_model codellama/CodeLlama-7b-hf \
  --data_path Data/LLMtrainDataset.jsonl \
  --output_dir ./LoraCodeLlama_7B \
  --batch_size 32 \
  --micro_batch_size 2 \
  --num_epochs 10 \
  --learning_rate 1e-4 \
  --cutoff_len 2048 \
  --val_set_size 100 \
  --prompt_template_name llama \
  --label_smoothing_factor 0.1 \
  --classification_alpha 0.5 \
  --train_on_inputs False
```

### **Option 2 – Multi-GPU (DistributedDataParallel)**

```bash
bash run-ddp.sh
```

`run-ddp.sh` uses:

```bash
torchrun --nproc_per_node=2 --master_port=29501 run-train.sh
```

Adjust `--nproc_per_node` to the number of GPUs available.

---

## 🧪 Testing

Once trained, evaluate the model with:

```bash
python -u test.py \
  --base_model codellama/CodeLlama-7b-hf \
  --lora_weights ./LoraCodeLlama_7B \
  --prompt_template llama
```

---

## 📌 Notes

* The first time you run training, model weights will be downloaded from Hugging Face and cached locally.
* To avoid tokenizer parallelism warnings, you can set:

```bash
export TOKENIZERS_PARALLELISM=false
```

* `train.log` is generated automatically during training; it is **ignored by git**.

---
