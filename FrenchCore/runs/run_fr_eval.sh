#!/bin/bash

cd ~/nanochat
source .venv/bin/activate

# Ajout des configurations d'environnement indispensables
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_lr_div6"
export PYTHONPATH=$(pwd):$PYTHONPATH

# Lancement du script avec les flags validés par ton argparse
python3 -m scripts.french_eval \
  -i sft \
  -g d12 \
  -s 1938 \
  -d ~/Main_CORE-fr/eval_data_fr
