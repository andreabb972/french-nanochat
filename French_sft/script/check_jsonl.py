import json

filepath = "/home/abinetru/French_sft/data/croissant_nanochat.jsonl"
print(f"🔍 Début de l'analyse de {filepath}...")

corrupted_count = 0
with open(filepath, "r") as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            messages = data.get("messages")
            if not isinstance(messages, list):
                corrupted_count += 1
                if corrupted_count <= 5:  # On affiche seulement les 5 premières pour ne pas surcharger
                    print(f"❌ Ligne {i} corrompue ! Type trouvé : {type(messages)}")
                    print(f"Contenu : {line[:200]}...\n")
        except Exception as e:
            corrupted_count += 1
            if corrupted_count <= 5:
                print(f"💥 Ligne {i} impossible à parser en JSON : {e}")

print(f"📋 Analyse terminée. Nombre total de lignes problématiques : {corrupted_count}")
