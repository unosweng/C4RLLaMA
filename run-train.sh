#!/bin/bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate c4rllama

export CUDA_VISIBLE_DEVICES=0,1

python -u train.py --base_model codellama/CodeLlama-7b-hf \
  --data_path Data/LLMtrainDataset.jsonl --output_dir ./LoraCodeLlama_7B \
  --batch_size 32 --micro_batch_size 2 --num_epochs 10 --learning_rate 1e-4 \
  --cutoff_len 2048 --val_set_size 100 --prompt_template_name llama \
  --label_smoothing_factor 0.1 --classification_alpha 0.5 --train_on_inputs False
