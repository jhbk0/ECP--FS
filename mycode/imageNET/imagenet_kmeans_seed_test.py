# Test K-means seeds for ImageNet Level 2, capacity 128.
from pathlib import Path
from time import perf_counter
from datetime import datetime
import csv
import gc
import math

import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Capacity means target vectors per cluster.
LEVEL = 2
CAPACITY = 128
SEEDS = [10, 21, 100, 123]  # Seed 42 failed previously.
BATCH_SIZE = 4096

HERE = Path(__file__).resolve().parent
NAME = "imagenet-clip-512-normalized.h5"
DATA_OPTIONS = [
    HERE / "data" / NAME,
    HERE.parent.parent / "data" / NAME,
    HERE.parent.parent.parent / "data" / NAME,
    Path.cwd() / "data" / NAME,
]
DATA = next((p for p in DATA_OPTIONS if p.is_file()), None)
if DATA is None:
    raise FileNotFoundError("Dataset not found. Checked:\n" + "\n".join(map(str, DATA_OPTIONS)))

OUTPUT = HERE / "imagenet_final_output"
PREPARED = OUTPUT / "prepared"
PREPARED.mkdir(parents=True, exist_ok=True)
LOG = OUTPUT / "mbk_seed_test.csv"
if not LOG.exists():
    with LOG.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "time", "dataset", "level", "capacity_vectors", "seed",
            "clusters_requested", "empty_groups", "seconds", "status", "details"
        ])


def record(seed, clusters, empty, seconds, status, details=""):
    with LOG.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"), "imagenet", LEVEL,
            CAPACITY, seed, clusters, empty, round(seconds, 2), status, details
        ])


print("Loading ImageNet vectors...", flush=True)
with h5py.File(DATA, "r") as f:
    train = np.asarray(f["train"], dtype=np.float32)

n_clusters = math.ceil(len(train) / CAPACITY)
print(f"Vectors: {len(train):,} | Dimensions: {train.shape[1]} | K-means groups: {n_clusters:,}", flush=True)

for seed in SEEDS:
    print(f"Testing seed {seed}...", flush=True)
    start = perf_counter()
    try:
        # Same K-means settings as the final ImageNet script.
        model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed,
            batch_size=BATCH_SIZE, n_init=1
        )
        model.fit(train)

        # Count assigned vectors in each K-means group.
        counts = np.bincount(model.labels_, minlength=n_clusters)
        empty_ids = np.flatnonzero(counts == 0)
        elapsed = perf_counter() - start
        if len(empty_ids):
            record(seed, n_clusters, len(empty_ids), elapsed, "empty_groups",
                   "First empty group: " + str(int(empty_ids[0])))
            print(f"Seed {seed}: {len(empty_ids)} empty group(s), first ID {empty_ids[0]} ({elapsed:.1f}s)", flush=True)
            del model
            gc.collect()
            continue

        # Save representatives so successful preparation is not repeated.
        print("No empty groups. Saving representatives...", flush=True)
        chosen = []
        for c in range(n_clusters):
            members = np.flatnonzero(model.labels_ == c)
            distances = np.sum((train[members] - model.cluster_centers_[c]) ** 2, axis=1)
            chosen.append(int(members[np.argmin(distances)]))

        ids = np.sort(np.asarray(chosen, dtype=np.uint32))
        vectors = np.ascontiguousarray(train[ids], dtype=np.float32)
        prep_file = PREPARED / f"imagenet_kmeans_C{CAPACITY}_seed{seed}.npz"
        if prep_file.exists():
            print(f"Prepared file already exists; not overwriting: {prep_file}", flush=True)
        else:
            np.savez(prep_file, ids=ids, vectors=vectors)
        elapsed = perf_counter() - start
        record(seed, n_clusters, 0, elapsed, "success", str(prep_file))
        print(f"SUCCESS: seed {seed} ({elapsed:.1f}s)", flush=True)
        print("Prepared file:", prep_file)
        print(f"For the full ImageNet K-means script, set SEED = {seed}.")
        break
    except MemoryError:
        elapsed = perf_counter() - start
        record(seed, n_clusters, "", elapsed, "memory_error")
        print("Out of memory. Stop here; do not continue other seeds.", flush=True)
        raise
    except Exception as e:
        elapsed = perf_counter() - start
        record(seed, n_clusters, "", elapsed, "error", str(e))
        print(f"Seed {seed}: {type(e).__name__}: {e}", flush=True)
    finally:
        if "model" in locals():
            del model
        gc.collect()
else:
    print("All tested seeds failed. Mark ImageNet L2/C128 as failed_mbk for now.", flush=True)

print("Seed test log:", LOG)
