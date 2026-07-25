import pyarrow.parquet as pq
import pyarrow as pa
import glob
import numpy as np
import os

INPUT_DIR = os.environ.get('INPUT_DIR', '.')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
N_OUTPUT = 34
SEED = 42
BUFFER_SIZE = 50000

rng = np.random.default_rng(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ETAPE 1 : Distribuer directement dans les 34 fichiers output
# ============================================================
print("=== ETAPE 1 : Distribution dans les fichiers output ===", flush=True)

output_paths = [os.path.join(OUTPUT_DIR, f'train-{i:05d}-of-{N_OUTPUT:05d}.parquet') for i in range(N_OUTPUT)]
writers = [None] * N_OUTPUT
buffers = [[] for _ in range(N_OUTPUT)]

def flush_buffer(i, writers, path, buffer):
    table = pa.concat_tables(buffer)
    if writers[i] is None:
        writers[i] = pq.ParquetWriter(path, table.schema)
    writers[i].write_table(table)
    return []

files = sorted(glob.glob(os.path.join(INPUT_DIR, 'train-*.parquet')))
print(f"Nombre de fichiers: {len(files)}", flush=True)

for i, f in enumerate(files):
    table = pq.read_table(f)
    n = len(table)
    output_ids = rng.integers(0, N_OUTPUT, size=n)
    for o in range(N_OUTPUT):
        mask = output_ids == o
        if mask.sum() > 0:
            buffers[o].append(table.filter(mask))
    for o in range(N_OUTPUT):
        if sum(len(t) for t in buffers[o]) >= BUFFER_SIZE:
            buffers[o] = flush_buffer(o, writers, output_paths[o], buffers[o])
    print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} — {n} documents distribués", flush=True)

# Flush final
for o in range(N_OUTPUT):
    if buffers[o]:
        flush_buffer(o, writers, output_paths[o], buffers[o])
for o in range(N_OUTPUT):
    if writers[o]:
        writers[o].close()

print("Distribution terminée !", flush=True)

# ============================================================
# ETAPE 2 : Shuffler chaque fichier output
# ============================================================
print("\n=== ETAPE 2 : Shuffle de chaque fichier output ===", flush=True)

def cast_to_large_string(table):
    new_fields, new_columns = [], []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if field.type == pa.string():
            col = col.cast(pa.large_string())
            field = field.with_type(pa.large_string())
        new_fields.append(field)
        new_columns.append(col)
    return pa.table(new_columns, schema=pa.schema(new_fields))

for i, path in enumerate(output_paths):
    print(f"  [{i+1}/{N_OUTPUT}] {os.path.basename(path)}: lecture...", flush=True)
    table = pq.read_table(path)
    table = cast_to_large_string(table)
    table = table.take(rng.permutation(len(table)))
    pq.write_table(table, path)
    print(f"  [{i+1}/{N_OUTPUT}] {os.path.basename(path)}: {len(table)} documents shufflés", flush=True)
    del table

print("\nDone!", flush=True)
