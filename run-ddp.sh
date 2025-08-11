#!/bin/bash

# HOW TO RUN
# 
# nohup ./run-ddp.sh > train.log 2>&1 &

set -euo pipefail

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate c4rllama

# GPUs & threading
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
# If you hit NCCL issues, uncomment:
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# Start distributed training (2 GPUs)
exec torchrun --nproc_per_node=2 --master_port=29501 \
  train.py \
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

