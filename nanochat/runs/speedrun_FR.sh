#!/bin/bash

cd /home/abinetru/nanochat
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=$(pwd):$PYTHONPATH

# Installation/Vérification de uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Activation du bon environnement virtuel
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "/home/abinetru/.venv" ]; then
    source /home/abinetru/.venv/bin/activate
else
    uv venv && source .venv/bin/activate
fi

# Synchronisation des dépendances GPU
uv sync --extra gpu

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_H100_50_tok"
mkdir -p $NANOCHAT_BASE_DIR
export OMP_NUM_THREADS=1
DATASET_75B_DIR="$HOME/.cache/huggingface/hub/datasets--manu--NanoCroissant-75B/data"

rm -rf "$NANOCHAT_BASE_DIR/data" "$NANOCHAT_BASE_DIR/base_data"

# On crée les deux liens symboliques pour satisfaire tous les scripts de NanoChat
ln -s "$DATASET_75B_DIR" "$NANOCHAT_BASE_DIR/data"
ln -s "$DATASET_75B_DIR" "$NANOCHAT_BASE_DIR/base_data"

export WANDB_API_KEY="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"
WANDB_RUN="croissant_H100_150tok"


# Initialisation du rapport de performance
python3 -m nanochat.report reset

echo "Entraînement du Tokenizer..."
python3 -m scripts.tok_train

# Validation du Tokenizer
python3 -m scripts.tok_eval

# --- Pretraining ---
echo "Lancement du Pretraining sur le dataset Croissant-75B..."
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --window-pattern=L \
    --device-batch-size=16 \
    --run=$WANDB_RUN

# Évaluation du modèle de base
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=16

# --- SFT (Finetuning supervisé) ---
echo "Lancement du SFT Supervisé avec chat_fr_sft"

# Téléchargement des conversations d'identité
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl h0ttps://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Exécution de ton script SFT personnalisé
torchrun --standalone --nproc_per_node=1 -m scripts.chat_fr_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN

# Évaluation finale du Chat
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# Génération du rapport de performances final
python3 -m nanochat.report generate
echo "Speedrun terminé ! Tous tes fichiers sont stockés en sécurité dans $NANOCHAT_BASE_DIR"
