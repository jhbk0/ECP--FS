# Import libraries
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from ecpfs import Builder, Index, Metric


# Load CCNews
DATA = Path("../../data/ccnews-nomic-768-normalized.h5")
RESULTS = Path("results")
INDEXES = Path("indexes")
RESULTS.mkdir(exist_ok=True)
INDEXES.mkdir(exist_ok=True)

with h5py.File(DATA, "r") as f:
    queries = np.asarray(f["test"], dtype=np.float32)
    neighbors = f["neighbors"][:]


# Parameters
levels_list = [2, 3, 4]
capacities = [128, 256, 512]
expansions = [16, 32, 64, 100]
k_values = [10, 30, 50, 100]


# K-means representatives
with h5py.File(DATA, "r") as f:
    train = np.asarray(f["train"], dtype=np.float32)

representatives = {}
preparation_times = {}

for capacity in capacities:
    n_clusters = int(np.ceil(len(train) / capacity))
    start = perf_counter()
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=1)
    model.fit(train)

    selected_ids = []
    for cluster in range(n_clusters):
        members = np.flatnonzero(model.labels_ == cluster)
        if len(members) == 0:
            raise ValueError("K-means has an empty cluster")
        distances = np.sum((train[members] - model.cluster_centers_[cluster]) ** 2, axis=1)
        selected_ids.append(members[np.argmin(distances)])

    ids = np.sort(np.asarray(selected_ids, dtype=np.uint32))
    representatives[capacity] = (ids, np.ascontiguousarray(train[ids]))
    preparation_times[capacity] = perf_counter() - start
    print("K-means prepared", capacity, flush=True)

del train


# Build indexes
build_rows = []

for levels in levels_list:
    for capacity in capacities:
        index_path = INDEXES / f"ccnews_kmeans_l{levels}_c{capacity}.zarr"
        print("Building", index_path.name, flush=True)

        builder = Builder(
            index_path=str(index_path),
            levels=levels,
            metric=Metric.IP,
            is_normalized=True,
        )

        ids, vectors = representatives[capacity]
        start = perf_counter()
        builder.select_representatives_custom(ids, vectors)
        selection_time = perf_counter() - start

        start = perf_counter()
        builder.build(
            embeddings_file=str(DATA),
            fallback_batch_rows=100_000,
            grp_name="train",
        )
        build_time = perf_counter() - start

        build_rows.append({
            "levels": levels,
            "capacity": capacity,
            "representatives": len(ids),
            "kmeans_time_s": preparation_times[capacity],
            "selection_time_s": selection_time,
            "build_time_s": build_time,
        })
        pd.DataFrame(build_rows).to_csv(RESULTS / "kmeans_build.csv", index=False)


# Search parameters
search_rows = []

for levels in levels_list:
    for capacity in capacities:
        index_path = INDEXES / f"ccnews_kmeans_l{levels}_c{capacity}.zarr"
        for expansion in expansions:
            for k in k_values:
                with Index(str(index_path)) as index:
                    for pass_name in ["first", "warm"]:
                        latencies = []
                        hits = 0
                        pass_start = perf_counter()

                        for i, query in enumerate(queries):
                            start = perf_counter()
                            items, query_id = index.new_search(
                                query=query,
                                k=k,
                                search_exp=expansion,
                                max_increments=0,
                                exclude_vec=[],
                            )
                            latencies.append((perf_counter() - start) * 1000)
                            index.end_search(query_id)

                            found = {item_id for _, item_id in items}
                            hits += len(found & set(neighbors[i, :k]))

                        elapsed = perf_counter() - pass_start

                        search_rows.append({
                            "levels": levels,
                            "capacity": capacity,
                            "expansion": expansion,
                            "k": k,
                            "pass": pass_name,
                            "latency_ms": float(np.mean(latencies)),
                            "recall": hits / (len(queries) * k),
                            "qps": len(queries) / elapsed,
                        })

                # Save results
                pd.DataFrame(search_rows).to_csv(RESULTS / "kmeans_search.csv", index=False)

print("Finished kmeans")

