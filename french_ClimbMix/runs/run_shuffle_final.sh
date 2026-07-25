#!/bin/bash
set -e

echo "Démarrage: $(date)"
python3 ~/french_ClimbMix/shuffle_final.py
echo "Fin: $(date)"
