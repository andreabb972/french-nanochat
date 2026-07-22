import json
from datasets import load_dataset
import glob
import copy

# 1. Chargement des fichiers parquet de CroissantLLM
parquet_files = glob.glob("./croissant_raw/data/train-*.parquet")
parquet_files.sort()
print(f"Fichiers trouvés : {parquet_files}")

ds = load_dataset("parquet", data_files=parquet_files, split="train")
print(f"Total exemples : {len(ds)}")

role_map = {"human": "user", "gpt": "assistant", "system": "system"}
skipped = 0
written = 0

with open("croissant_nanochat.jsonl", "w", encoding="utf-8") as f_out:
    for row in ds:
        conversation = row["conversations"]
        messages = []
        
        
        for turn in conversation:
            role = role_map.get(turn.get("from"))
            content = turn.get("value") or turn.get("text")
            if role and content and content.strip():
                if messages and messages[-1]["role"] == role:
                    messages[-1]["content"] += "\n\n" + content.strip()
                else:
                    messages.append({"role": role, "content": content.strip()})

        if not messages:
            skipped += 1
            continue

        test_messages = copy.deepcopy(messages)
        if test_messages[0]["role"] == "system":
            if len(test_messages) < 2 or test_messages[1]["role"] != "user":
                skipped += 1
                continue
            test_messages = test_messages[1:]

        valid_alternation = True
        for i, msg in enumerate(test_messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            if msg["role"] != must_be_from:
                valid_alternation = False
                break
        
        if not valid_alternation:
            skipped += 1
            continue

        if test_messages[-1]["role"] != "assistant":
            messages.pop()

        if len(messages) < 2:
            skipped += 1
            continue
        json_line = {"messages": messages}
        f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        written += 1

print(f"\n Écrit    : {written} conversations")
print(f"  : {skipped} conversations")
print(f"Fichier  : croissant_nanochat.jsonl")
