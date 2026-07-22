#!/bin/bash

cd /home/abinetru/nanochat
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=$(pwd):$PYTHONPATH

export WANDB_API_KEY="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"
export WANDB_RUN="nanochat_sft_div6"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_lr_div6/"

source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 -m scripts.chat_fr_sft -- \
  --device-batch-size=16 \
  --embedding-lr 0.0333 \
  --unembedding-lr 0.000445 \
  --matrix-lr 0.00222 \
  --run=$WANDB_RUN
# Évaluation du modèle SFT après son entraînement
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft
# Génération du rapport final
python -m nanochat.report generate
