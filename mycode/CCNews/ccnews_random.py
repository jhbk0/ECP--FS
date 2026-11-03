# Import libraries
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
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

strategy = "random"


# Build indexes
build_rows = []

for levels in levels_list:
    for capacity in capacities:
        index_path = INDEXES / f"ccnews_{strategy}_l{levels}_c{capacity}.zarr"
        print("Building", index_path.name, flush=True)

        builder = Builder(
            index_path=str(index_path),
            levels=levels,
            metric=Metric.IP,
            is_normalized=True,
        )

        start = perf_counter()
        builder.select_representatives(
            embeddings_file=str(DATA),
            target_cluster_items=capacity,
            strategy=strategy,
            fallback_batch_rows=100_000,
            grp_name="train",
        )
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
            "selection_time_s": selection_time,
            "build_time_s": build_time,
        })
        pd.DataFrame(build_rows).to_csv(RESULTS / f"{strategy}_build.csv", index=False)


# Search parameters
search_rows = []

for levels in levels_list:
    for capacity in capacities:
        index_path = INDEXES / f"ccnews_{strategy}_l{levels}_c{capacity}.zarr"
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
                pd.DataFrame(search_rows).to_csv(RESULTS / f"{strategy}_search.csv", index=False)

print("Finished", strategy)

