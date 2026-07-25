import pyarrow.parquet as pq
import pyarrow as pa
import glob
import numpy as np
import os
import shutil

INPUT_DIR = os.environ.get('INPUT_DIR', '.')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
TEMP_DIR = os.environ.get('TEMP_DIR', './tmp_buckets')
N_BUCKETS = 20
SEED = 42

rng = np.random.default_rng(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print("=== ETAPE 1 : Distribution dans les buckets ===", flush=True)

bucket_paths = [os.path.join(TEMP_DIR, f'bucket_{b:02d}.parquet') for b in range(N_BUCKETS)]
writers = [None] * N_BUCKETS
bucket_buffers = [[] for _ in range(N_BUCKETS)]
BUFFER_SIZE = 50000

def flush_buffer(b, writers, path, buffer):
    table = pa.concat_tables(buffer)
    if writers[b] is None:
        writers[b] = pq.ParquetWriter(path, table.schema)
    writers[b].write_table(table)
    return []

files = sorted(glob.glob(os.path.join(INPUT_DIR, 'train-*.parquet')))
print(f"Nombre de fichiers: {len(files)}", flush=True)

for i, f in enumerate(files):
    table = pq.read_table(f)
    n = len(table)
    bucket_ids = rng.integers(0, N_BUCKETS, size=n)
    for b in range(N_BUCKETS):
        mask = bucket_ids == b
        if mask.sum() > 0:
            bucket_buffers[b].append(table.filter(mask))
    for b in range(N_BUCKETS):
        if sum(len(t) for t in bucket_buffers[b]) >= BUFFER_SIZE:
            bucket_buffers[b] = flush_buffer(b, writers, bucket_paths[b], bucket_buffers[b])
    print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} — {n} documents distribués", flush=True)

for b in range(N_BUCKETS):
    if bucket_buffers[b]:
        flush_buffer(b, writers, bucket_paths[b], bucket_buffers[b])
for b in range(N_BUCKETS):
    if writers[b]:
        writers[b].close()

print("Distribution terminée !", flush=True)
for b, path in enumerate(bucket_paths):
    size_gb = os.path.getsize(path) / 1e9
    print(f"  Bucket {b:02d}: {size_gb:.1f} Go", flush=True)

print("\n=== ETAPE 2 : Shuffle et écriture par bucket ===", flush=True)

def cast_to_large_string(table):
    new_fields = []
    new_columns = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if field.type == pa.string():
            col = col.cast(pa.large_string())
            field = field.with_type(pa.large_string())
        new_fields.append(field)
        new_columns.append(col)
    return pa.table(new_columns, schema=pa.schema(new_fields))

files_per_bucket = 130 // N_BUCKETS
extra = 130 % N_BUCKETS
output_file_idx = 0

for b, path in enumerate(bucket_paths):
    print(f"\n  Bucket {b:02d}: lecture...", flush=True)
    table = pq.read_table(path)
    table = cast_to_large_string(table)
    print(f"  Bucket {b:02d}: {len(table)} documents — shuffle...", flush=True)
    table = table.take(rng.permutation(len(table)))
    n_files = files_per_bucket + (1 if b < extra else 0)
    chunk_size = len(table) // n_files
    for j in range(n_files):
        start = j * chunk_size
        end = start + chunk_size if j < n_files - 1 else len(table)
        chunk = table.slice(start, end - start)
        out = os.path.join(OUTPUT_DIR, f'train-{output_file_idx:05d}-of-00130.parquet')
        pq.write_table(chunk, out)
        print(f"    [{output_file_idx+1}/130] {os.path.basename(out)} — {len(chunk)} documents", flush=True)
        output_file_idx += 1
    del table

print("\nNettoyage des buckets temporaires...", flush=True)
shutil.rmtree(TEMP_DIR)
print(f"\nDone! {output_file_idx} fichiers écrits.", flush=True)
