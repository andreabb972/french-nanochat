#!/bin/bash

echo "📁 Création de l'arborescence du dossier French_sft..."
# Crée les dossiers nécessaires s'ils n'existent pas
mkdir -p French_sft/data

echo "⏳ Lancement du script de conversion..."
echo "----------------------------------------"

# Exécute ton script python (assure-toi que convert_croissant.py est dans le même dossier)
python convert_croissant.py

echo "----------------------------------------"
# Vérification de la création du fichier
if [ -f "croissant_nanochat.jsonl" ]; then
    echo "📦 Déplacement du fichier converti vers French_sft/data/..."
    mv croissant_nanochat.jsonl French_sft/data/
    echo "✅ Terminé avec succès !"
    echo "📄 Ton dataset est prêt ici : French_sft/data/croissant_nanochat.jsonl"
else
    echo "❌ Erreur : Le fichier croissant_nanochat.jsonl n'a pas été généré."
fi
