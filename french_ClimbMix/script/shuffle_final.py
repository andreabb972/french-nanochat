import pyarrow.parquet as pq
import pyarrow as pa
import glob
import numpy as np
import os

ENGLISH_DIR = os.path.expanduser("~/.cache/huggingface/hub/datasets--manu--english-30b-shuffled/data")
FRENCH_DIR  = os.path.expanduser("~/.cache/huggingface/hub/datasets--manu--french-30b/snapshots/6bdeaf311e673575f0f7d4c7f90d7ffbd879729b/data")
CODE_DIR    = os.path.expanduser("~/.cache/huggingface/hub/datasets--manu--code_20b-shuffled/data")
OUTPUT_DIR  = os.path.expanduser("~/.cache/huggingface/hub/datasets--manu--NanoCroissant-75B/data")
N_OUTPUT = 100
SEED = 42
BUFFER_SIZE = 50000

rng = np.random.default_rng(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cast_to_large_string(table):
    # Ordre canonique des colonnes
    COLUMNS = ['id', 'text', 'dataset_id']
    table = table.select(COLUMNS)
    new_fields, new_columns = [], []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if field.type == pa.string():
            col = col.cast(pa.large_string())
            field = field.with_type(pa.large_string())
        new_fields.append(field)
        new_columns.append(col)
    return pa.table(new_columns, schema=pa.schema(new_fields))

english_files = sorted(glob.glob(os.path.join(ENGLISH_DIR, 'train-*.parquet')))[:65]
french_files  = sorted(glob.glob(os.path.join(FRENCH_DIR,  'train-*.parquet')))
code_files    = sorted(glob.glob(os.path.join(CODE_DIR,    'train-*.parquet')))[:25]

all_files = english_files + french_files + code_files
print(f"Fichiers input: {len(english_files)} anglais + {len(french_files)} français + {len(code_files)} code = {len(all_files)} total", flush=True)


print("\n=== ETAPE 1 : Distribution dans les fichiers output ===", flush=True)

output_paths = [os.path.join(OUTPUT_DIR, f'train-{i:05d}-of-{N_OUTPUT:05d}.parquet') for i in range(N_OUTPUT)]
writers = [None] * N_OUTPUT
buffers = [[] for _ in range(N_OUTPUT)]

def flush_buffer(i, writers, path, buffer):
    table = pa.concat_tables(buffer)
    if writers[i] is None:
        writers[i] = pq.ParquetWriter(path, table.schema)
    writers[i].write_table(table)
    return []

for i, f in enumerate(all_files):
    table = pq.read_table(f)
    table = cast_to_large_string(table)
    n = len(table)
    output_ids = rng.integers(0, N_OUTPUT, size=n)
    for o in range(N_OUTPUT):
        mask = output_ids == o
        if mask.sum() > 0:
            buffers[o].append(table.filter(mask))
    for o in range(N_OUTPUT):
        if sum(len(t) for t in buffers[o]) >= BUFFER_SIZE:
            buffers[o] = flush_buffer(o, writers, output_paths[o], buffers[o])
    print(f"  [{i+1}/{len(all_files)}] {os.path.basename(f)} — {n} documents distribués", flush=True)

for o in range(N_OUTPUT):
    if buffers[o]:
        flush_buffer(o, writers, output_paths[o], buffers[o])
for o in range(N_OUTPUT):
    if writers[o]:
        writers[o].close()

print("Distribution terminée !", flush=True)
print("\n=== ETAPE 2 : Shuffle de chaque fichier output ===", flush=True)

for i, path in enumerate(output_paths):
    print(f"  [{i+1}/{N_OUTPUT}] {os.path.basename(path)}: lecture...", flush=True)
    table = pq.read_table(path)
    table = table.take(rng.permutation(len(table)))
    pq.write_table(table, path)
    print(f"  [{i+1}/{N_OUTPUT}] {os.path.basename(path)}: {len(table)} documents shufflés", flush=True)
    del table

print("\nDone!", flush=True)
