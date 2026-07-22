#!/bin/bash
cd ~/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p $NANOCHAT_BASE_DIR

source .venv/bin/activate

# télécharger les identity conversations si pas déjà fait
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# SFT avec CroissantLLM + MMLU + GSM8K
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run dummy

# eval
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft
