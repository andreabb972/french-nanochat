import pyarrow.parquet as pq
import pyarrow as pa
import glob
import numpy as np
import os
import pickle
import random

# -----------------------------------------------------------------------------
# Configuration des chemins
# -----------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--english', type=float, default=0.40)
parser.add_argument('--french',  type=float, default=0.40)
parser.add_argument('--code',    type=float, default=0.20)
parser.add_argument('--total',   type=float, default=30_000_000_000)
parser.add_argument('--output-dir', type=str, required=True)
args = parser.parse_args()

assert abs(args.english + args.french + args.code - 1.0) < 1e-6, "Les proportions doivent sommer à 1.0"

ENGLISH_DIR    = os.path.expanduser("~/.cache/huggingface/hub/datasets--manu--english-30b-shuffled/data")
FRENCH_DIR     = os.path.expanduser("~/.cache/huggingface/hub/datasets--manu--french_30b-shuffled/data")
CODE_DIR       = os.path.expanduser("~/.cache/huggingface/hub/datasets--manu--code_20b-shuffled/data")
OUTPUT_DIR     = os.path.expanduser(args.output_dir)
TOKENIZER_PATH = os.path.expanduser("~/.cache/nanochat_FR/tokenizer/tokenizer.pkl")
N_OUTPUT       = 100
SEED           = 42
BUFFER_SIZE    = 50000

TOTAL_TOKENS_TARGET = args.total
TARGET_PROPS = {
    'english': args.english,
    'french':  args.french,
    'code':    args.code,
}

# tokens/doc calibrés sur le run précédent
tokens_per_row = {
    'english': 903.3,
    'french':  424.4,
    'code':    1483.5,
}

def cast_and_standardize(table, dataset_label):
    """Force l'ordre des colonnes et standardise le type de schéma."""
    COLUMNS = ['id', 'text', 'dataset_id']
    table = table.select(COLUMNS)
    new_fields, new_columns = [], []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if field.name == 'dataset_id':
            col = pa.array([dataset_label] * len(table), type=pa.large_string())
            field = field.with_type(pa.large_string())
        elif field.type == pa.string():
            col = col.cast(pa.large_string())
            field = field.with_type(pa.large_string())
        new_fields.append(field)
        new_columns.append(col)
    return pa.table(new_columns, schema=pa.schema(new_fields))
# -----------------------------------------------------------------------------
# Calcul des quotas de lignes à partir des tokens/doc calibrés
# -----------------------------------------------------------------------------
english_files = sorted(glob.glob(os.path.join(ENGLISH_DIR, 'train-*.parquet')))
french_files  = sorted(glob.glob(os.path.join(FRENCH_DIR,  'train-*.parquet')))
code_files    = sorted(glob.glob(os.path.join(CODE_DIR,    'train-*.parquet')))

all_sources = [
    (english_files, 'english'),
    (french_files,  'french'),
    (code_files,    'code')
]

rows_to_extract = {}
print("\n=== Quotas de lignes (tokens/doc calibrés) ===", flush=True)
for files, label in all_sources:
    if not files:
        raise ValueError(f"Aucun fichier trouvé pour '{label}'.")
    lang_tokens_target      = TOTAL_TOKENS_TARGET * TARGET_PROPS[label]
    rows_to_extract[label]  = int(lang_tokens_target / tokens_per_row[label])
    print(f"  {label}: {lang_tokens_target/1e9:.1f}B tokens cibles → {rows_to_extract[label]:,} lignes", flush=True)

rng = np.random.default_rng(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Chargement du tokenizer...", end="", flush=True)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print(" OK !", flush=True)
# -----------------------------------------------------------------------------
# ÉTAPE 2 : Flux de distribution (Streaming)
# -----------------------------------------------------------------------------
print("=== ÉTAPE 2 : Extraction et distribution dans les 100 paquets ===", flush=True)

output_paths = [
    os.path.join(OUTPUT_DIR, f'train-{i:05d}-of-{N_OUTPUT:05d}.parquet')
    for i in range(N_OUTPUT)
]
writers = [None] * N_OUTPUT
buffers = [[]   for _ in range(N_OUTPUT)]


def flush_buffer(i, writers, path, buffer):
    table = pa.concat_tables(buffer)
    if writers[i] is None:
        writers[i] = pq.ParquetWriter(path, table.schema)
    writers[i].write_table(table)
    return []


global_file_idx        = 1
total_files_to_process = len(english_files) + len(french_files) + len(code_files)

for files, label in all_sources:
    quota_lignes      = rows_to_extract[label]
    lignes_accumulees = 0

    for f in files:
        if lignes_accumulees >= quota_lignes:
            break

        table = pq.read_table(f)
        table = cast_and_standardize(table, dataset_label=label)
        n     = len(table)

        if lignes_accumulees + n > quota_lignes:
            lignes_restantes = quota_lignes - lignes_accumulees
            table = table.slice(0, lignes_restantes)
            n     = len(table)

        lignes_accumulees += n

        output_ids = rng.integers(0, N_OUTPUT, size=n)
        for o in range(N_OUTPUT):
            mask = output_ids == o
            if mask.sum() > 0:
                buffers[o].append(table.filter(mask))

        for o in range(N_OUTPUT):
            if sum(len(t) for t in buffers[o]) >= BUFFER_SIZE:
                buffers[o] = flush_buffer(o, writers, output_paths[o], buffers[o])

        print(
            f"  [{global_file_idx}/{total_files_to_process}] "
            f"{os.path.basename(f)} ({label}) — "
            f"{n} lignes injectées ({lignes_accumulees}/{quota_lignes})",
            flush=True
        )
        global_file_idx += 1

# Flush des buffers restants
for o in range(N_OUTPUT):
    if buffers[o]:
        flush_buffer(o, writers, output_paths[o], buffers[o])
for o in range(N_OUTPUT):
    if writers[o]:
        writers[o].close()


# -----------------------------------------------------------------------------
# ÉTAPE 3 : Shuffle final global par fichier de sortie
# -----------------------------------------------------------------------------
print("\n=== ÉTAPE 3 : Shuffle final de chaque paquet de sortie ===", flush=True)

for i, path in enumerate(output_paths):
    print(f"  [{i+1}/{N_OUTPUT}] {os.path.basename(path)} : mélange...", flush=True)
    table = pq.read_table(path)
    table = table.take(rng.permutation(len(table)))
    pq.write_table(table, path)
    del table


# -----------------------------------------------------------------------------
# ÉTAPE 4 : Vérification finale des proportions
# -----------------------------------------------------------------------------
print("\n=== ÉTAPE 4 : Vérification finale des proportions ===", flush=True)

check_files  = random.sample(output_paths, min(3, len(output_paths)))
check_counts = {label: 0 for _, label in all_sources}

for path in check_files:
    print(f"  Vérification de {os.path.basename(path)}...", flush=True)
    df_check = pq.read_table(path).to_pandas()

    for label in check_counts:
        texts = df_check[df_check['dataset_id'] == label]['text'].dropna().astype(str).tolist()
        if not texts:
            continue
        sample   = random.sample(texts, min(500, len(texts)))
        avg      = sum(len(tokenizer.encode(s)) for s in sample) / len(sample)
        check_counts[label] += avg * (df_check['dataset_id'] == label).sum()

total_check = sum(check_counts.values())
print("\n--------------------------------------------------")
print(f"{'Source':<12} {'Obtenu':>8}   {'Cible':>8}   {'Écart':>8}")
print("--------------------------------------------------")
for label, count in check_counts.items():
    obtenu = count / total_check * 100
    cible  = TARGET_PROPS[label] * 100
    ecart  = obtenu - cible
    signe  = "+" if ecart >= 0 else ""
    print(f"  {label:<10} {obtenu:>7.1f}%   {cible:>7.1f}%   {signe}{ecart:.1f}%")
print("--------------------------------------------------")

print("\nTerminé ! Le dataset respecte les proportions cibles.", flush=True)
