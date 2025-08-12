# run-test.sh
#!/bin/bash
set -euo pipefail
# load conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate c4rllama

export TOKENIZERS_PARALLELISM=false
# pick a GPU (or comment this out to use any)
export CUDA_VISIBLE_DEVICES=0

python -u test.py \
  --base_model codellama/CodeLlama-7b-hf \
  --lora_weights ./LoraCodeLlama_7B \
  --prompt_template llama
