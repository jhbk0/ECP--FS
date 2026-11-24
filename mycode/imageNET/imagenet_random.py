# ImageNet CLIP eCP-FS random benchmark
from pathlib import Path
from time import perf_counter, sleep
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
import csv
import gzip
import json
import math
import shutil
import threading
import traceback

import h5py
import numpy as np
import psutil
from ecpfs import Builder, Index, Metric


# Cluster capacity is the target number of vectors per cluster.
LEVELS = [2, 3]
CAPACITIES = [128, 256, 512, 1024]
EXPANSIONS = [16, 32, 64, 100, 128, 256, 512]
K_VALUES = [10, 30, 50, 100]
STRATEGY = "random"
WARM_RUNS = 5
FULL_LEAF_QUERIES = 25
SEED = 42
BATCH_ROWS = 100_000

# The formula capacities use 128 KiB and 1 MiB of raw vector data.
INCLUDE_FORMULA = False
FORMULA_LEVEL = 2

HERE = Path(__file__).resolve().parent
NAME = "imagenet-clip-512-normalized.h5"
DATA_OPTIONS = [
    HERE / "data" / NAME,
    HERE.parent.parent / "data" / NAME,
    HERE.parent.parent.parent / "data" / NAME,
    Path.cwd() / "data" / NAME,
    Path.cwd().parent / "data" / NAME,
]
DATA = next((p.resolve() for p in DATA_OPTIONS if p.is_file()), None)
if DATA is None:
    raise FileNotFoundError("ImageNet CLIP dataset not found. Checked:\n" + "\n".join(map(str, DATA_OPTIONS)))

OUTPUT = HERE / "imagenet_final_output"
RAW = OUTPUT / "raw"
INDEXES = OUTPUT / "indexes"
QUERY_FILES = OUTPUT / "query_runs"
COMPLETED = OUTPUT / "completed"
PREPARED = OUTPUT / "prepared"
for folder in [RAW, INDEXES, QUERY_FILES, COMPLETED, PREPARED]:
    folder.mkdir(parents=True, exist_ok=True)

# Files share index_id, search_config_id and search_run_id.
COLUMNS = {
    "capacity_plan.csv": ["index_id", "dataset", "strategy", "levels", "capacity_vectors", "capacity_source", "target_bytes", "dimensions", "dtype", "expected_leaf_count"],
    "preprocessing_runs.csv": ["dataset", "strategy", "capacity_vectors", "preparation_id", "seconds", "peak_rss_mib", "representatives", "seed"],
    "index_builds.csv": ["index_id", "dataset", "strategy", "levels", "capacity_vectors", "capacity_source", "representatives", "selection_s", "build_s", "preparation_id", "peak_selection_rss_mib", "peak_build_rss_mib", "index_size_mib", "actual_nonempty_leaves", "dataset_vectors", "ecpfs_version", "started_at", "status"],
    "index_nodes.csv": ["index_id", "level", "node_id", "kind", "vector_count", "id_count", "storage_bytes", "is_empty", "missing_embeddings", "over_capacity"],
    "cluster_summary.csv": ["index_id", "leaf_nodes", "nonempty_leaves", "empty_leaves", "missing_nodes", "total_leaf_vectors", "minimum", "median", "mean", "maximum", "p95", "std", "coefficient_of_variation", "largest_leaf_fraction", "over_capacity_leaves"],
    "search_plan.csv": ["search_config_id", "index_id", "search_type", "expansion", "k", "nonempty_leaves", "status", "reason"],
    "search_runs.csv": ["search_run_id", "search_config_id", "index_id", "strategy", "search_type", "expansion", "k", "pass_type", "repeat_no", "queries", "elapsed_search_s", "qps", "recall", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "peak_search_rss_mib", "hits", "time_recorded_at", "timing_method", "actual_leaves_visited"],
    "memory_samples.csv": ["index_id", "search_run_id", "phase", "seconds_from_start", "rss_mib", "available_ram_mib"],
    "events.csv": ["index_id", "time", "stage", "message"],
    "index_lifecycle.csv": ["index_id", "time", "event", "path"],
}
for filename, columns in COLUMNS.items():
    path = RAW / filename
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(columns)
    else:
        with path.open("r", encoding="utf-8") as file:
            if next(csv.reader(file)) != columns:
                raise ValueError("CSV header differs: " + str(path))


def add_row(filename, row):
    with (RAW / filename).open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS[filename])
        writer.writerow(row)


def read_rows(filename):
    with (RAW / filename).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def stamp():
    return datetime.now().isoformat(timespec="seconds")


def folder_size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# Sample RAM during builds and searches.
class Memory:
    def __init__(self, index_id, phase, search_run_id=""):
        self.index_id = index_id
        self.phase = phase
        self.search_run_id = search_run_id
        self.samples = []
        self.stop = threading.Event()

    def watch(self):
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                self.samples.append((perf_counter() - self.start,
                    process.memory_info().rss / 2**20,
                    psutil.virtual_memory().available / 2**20))
            except psutil.Error:
                break
            self.stop.wait(0.5)

    def __enter__(self):
        self.start = perf_counter()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        for seconds, ram, available in self.samples:
            add_row("memory_samples.csv", dict(index_id=self.index_id,
                search_run_id=self.search_run_id, phase=self.phase,
                seconds_from_start=round(seconds, 3), rss_mib=round(ram, 3),
                available_ram_mib=round(available, 3)))

    @property
    def peak(self):
        return max((row[1] for row in self.samples), default=0.0)


# Count vectors from the actual Zarr node metadata.
def inspect_index(path, index_id, levels, capacity, n_database):
    leaf_sizes = []
    empty = missing = over_capacity = leaf_nodes = 0
    for level in range(1, levels + 1):
        level_folder = path / f"lvl_{level}"
        if not level_folder.is_dir():
            raise RuntimeError(f"Missing level folder: {level_folder}")
        for node in sorted(level_folder.glob("node_*")):
            if not node.is_dir():
                continue
            emb = node / "embeddings"
            metadata = emb / "zarr.json"
            if not metadata.exists():
                metadata = emb / ".zarray"
            count = None
            if metadata.is_file():
                shape = json.loads(metadata.read_text(encoding="utf-8")).get("shape", [])
                if shape:
                    count = int(shape[0])
            ids_name = "item_ids" if level == levels else "node_ids"
            ids_folder = node / ids_name
            ids_meta = ids_folder / "zarr.json"
            if not ids_meta.exists():
                ids_meta = ids_folder / ".zarray"
            id_count = None
            if ids_meta.is_file():
                ids_shape = json.loads(ids_meta.read_text(encoding="utf-8")).get("shape", [])
                if ids_shape:
                    id_count = int(ids_shape[0])
            leaf = level == levels
            if leaf:
                leaf_nodes += 1
                if count is None or id_count != count:
                    missing += 1
                elif count == 0:
                    empty += 1
                else:
                    leaf_sizes.append(count)
                    if count > capacity:
                        over_capacity += 1
            add_row("index_nodes.csv", dict(index_id=index_id, level=level,
                node_id=node.name, kind="leaf" if leaf else "internal",
                vector_count=count if count is not None else "",
                id_count=id_count if id_count is not None else "",
                storage_bytes=folder_size(node),
                is_empty=int(leaf and count == 0),
                missing_embeddings=int(count is None),
                over_capacity=int(leaf and count is not None and count > capacity)))
    if not leaf_sizes or missing or sum(leaf_sizes) != n_database:
        raise RuntimeError(f"Leaf inspection incomplete: {sum(leaf_sizes)}/{n_database} vector records, {missing} invalid leaves")
    sizes = np.asarray(leaf_sizes, dtype=np.float64)
    average = float(np.mean(sizes))
    add_row("cluster_summary.csv", dict(index_id=index_id, leaf_nodes=leaf_nodes,
        nonempty_leaves=len(leaf_sizes), empty_leaves=empty, missing_nodes=missing,
        total_leaf_vectors=int(np.sum(sizes)), minimum=int(np.min(sizes)),
        median=float(np.median(sizes)), mean=average, maximum=int(np.max(sizes)),
        p95=float(np.percentile(sizes, 95)), std=float(np.std(sizes)),
        coefficient_of_variation=float(np.std(sizes) / average),
        largest_leaf_fraction=float(np.max(sizes) / n_database),
        over_capacity_leaves=over_capacity))
    return len(leaf_sizes)


# Save individual query measurements for every search pass.
def search_pass(index, index_id, config_id, expansion, k, kind, phase, repeat, queries, truth):
    run_id = f"{config_id}_{phase}{repeat}"
    query_file = QUERY_FILES / f"{run_id}.csv.gz"
    saved = {row["search_run_id"] for row in read_rows("search_runs.csv")}
    if run_id in saved:
        if not query_file.is_file():
            raise RuntimeError("Missing query CSV for completed run: " + run_id)
        return
    print("  ", kind, "exp", expansion, "k", k, phase, repeat, flush=True)
    timing = []
    results = []
    with Memory(index_id, "search", run_id) as memory:
        start_pass = perf_counter()
        for query in queries:
            start = perf_counter()
            items, query_id = index.new_search(query=query, k=k,
                search_exp=expansion, max_increments=0, exclude_vec=[])
            index.end_search(query_id)
            timing.append((perf_counter() - start) * 1000)
            results.append(items)
        elapsed = perf_counter() - start_pass

    hits = []
    counts = []
    for i, items in enumerate(results):
        found = {int(item_id) for _, item_id in items}
        hits.append(len(found & set(int(x) for x in truth[i, :k])))
        counts.append(len(found))
    temp = query_file.with_suffix(".tmp")
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["search_run_id", "query_id", "latency_ms", "hits_at_k", "returned_count"])
        for i, (ms, n_hits, returned) in enumerate(zip(timing, hits, counts)):
            writer.writerow([run_id, i, round(ms, 6), n_hits, returned])
    temp.replace(query_file)
    add_row("search_runs.csv", dict(search_run_id=run_id,
        search_config_id=config_id, index_id=index_id, strategy=STRATEGY,
        search_type=kind, expansion=expansion, k=k,
        pass_type=phase, repeat_no=repeat, queries=len(queries),
        elapsed_search_s=elapsed, qps=len(queries) / elapsed,
        recall=sum(hits) / (len(queries) * k),
        mean_latency_ms=float(np.mean(timing)),
        p50_latency_ms=float(np.percentile(timing, 50)),
        p95_latency_ms=float(np.percentile(timing, 95)),
        peak_search_rss_mib=memory.peak, hits=sum(hits),
        time_recorded_at=stamp(),
        timing_method="new_search+end_search; recall outside timer",
        actual_leaves_visited=""))


# Use 25 queries for the full-leaf pilot.
def test_index(index_path, index_id, leaf_count, queries, truth):
    configs = []
    planned = {row["search_config_id"] for row in read_rows("search_plan.csv")}
    for expansion in EXPANSIONS:
        for k in K_VALUES:
            config_id = f"{index_id}_E{expansion}_K{k}"
            allowed = expansion <= leaf_count
            if config_id not in planned:
                add_row("search_plan.csv", dict(search_config_id=config_id,
                    index_id=index_id, search_type="standard", expansion=expansion,
                    k=k, nonempty_leaves=leaf_count,
                    status="run" if allowed else "skipped",
                    reason="" if allowed else "Expansion > non-empty leaves"))
            if allowed:
                configs.append((config_id, expansion, k, "standard"))
    full_id = f"{index_id}_FULL_E{leaf_count}_K100"
    if full_id not in planned:
        add_row("search_plan.csv", dict(search_config_id=full_id,
            index_id=index_id, search_type="full_leaf_candidate", expansion=leaf_count,
            k=100, nonempty_leaves=leaf_count, status="run",
            reason="Actual leaf visits cannot be verified from the Python API"))
    configs.append((full_id, leaf_count, 100, "full_leaf_candidate"))

    for config_id, expansion, k, kind in configs:
        if kind == "standard":
            q, gt = queries, truth
            warm = WARM_RUNS
        else:
            q, gt = queries[:FULL_LEAF_QUERIES], truth[:FULL_LEAF_QUERIES]
            warm = 1
        with Index(str(index_path)) as index:
            search_pass(index, index_id, config_id, expansion, k, kind,
                "first", 0, q, gt)
            for repeat in range(1, warm + 1):
                search_pass(index, index_id, config_id, expansion, k, kind,
                    "warm", repeat, q, gt)

    # Check query CSVs before deleting the index.
    recorded = {row["search_run_id"]: row for row in read_rows("search_runs.csv")}
    for config_id, _, _, kind in configs:
        warm = WARM_RUNS if kind == "standard" else 1
        expected = len(queries) if kind == "standard" else min(FULL_LEAF_QUERIES, len(queries))
        for phase, rep in [("first", 0)] + [("warm", n) for n in range(1, warm + 1)]:
            run_id = f"{config_id}_{phase}{rep}"
            path = QUERY_FILES / f"{run_id}.csv.gz"
            if run_id not in recorded or not path.is_file():
                raise RuntimeError("Missing search result: " + run_id)
            with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) != expected:
                raise RuntimeError("Missing query records in: " + run_id)
            if sum(int(r["hits_at_k"]) for r in rows) != int(recorded[run_id]["hits"]):
                raise RuntimeError("Recall hits mismatch: " + run_id)


# Load all queries; use the existing normalized IP search.
with h5py.File(DATA, "r") as file:
    n_database, dim = file["train"].shape
    dtype = file["train"].dtype
    queries = np.asarray(file["test"], dtype=np.float32)
    neighbors = np.asarray(file["neighbors"], dtype=np.int64)
if neighbors.shape[1] < max(K_VALUES):
    raise ValueError("Ground truth must contain at least 100 neighbours")

try:
    ecp_version = version("ecpfs")
except PackageNotFoundError:
    ecp_version = "unknown"

capacities = [(level, cap, "standard", "")
              for level in LEVELS for cap in CAPACITIES]
if INCLUDE_FORMULA:
    raw_bytes_per_vector = dim * dtype.itemsize
    for source, target_bytes in [("formula_128KiB", 128 * 1024),
                                  ("formula_1MiB", 1024 * 1024)]:
        cap = max(1, target_bytes // raw_bytes_per_vector)
        if (FORMULA_LEVEL, cap, "standard", "") not in capacities:
            if not any(l == FORMULA_LEVEL and c == cap for l, c, _, _ in capacities):
                capacities.append((FORMULA_LEVEL, cap, source, target_bytes))
        plan_id = f"imagenet_{STRATEGY}_L{FORMULA_LEVEL}_C{cap}"
        if not any(r["index_id"] == plan_id and r["capacity_source"] == source
                   for r in read_rows("capacity_plan.csv")):
            add_row("capacity_plan.csv", dict(index_id=plan_id,
                dataset="imagenet", strategy=STRATEGY, levels=FORMULA_LEVEL,
                capacity_vectors=cap, capacity_source=source,
                target_bytes=target_bytes, dimensions=dim, dtype=str(dtype),
                expected_leaf_count=math.ceil(n_database / cap)))
for level in LEVELS:
    for cap in CAPACITIES:
        plan_id = f"imagenet_{STRATEGY}_L{level}_C{cap}"
        if not any(r["index_id"] == plan_id and r["capacity_source"] == "standard"
                   for r in read_rows("capacity_plan.csv")):
            add_row("capacity_plan.csv", dict(index_id=plan_id,
                dataset="imagenet", strategy=STRATEGY, levels=level,
                capacity_vectors=cap, capacity_source="standard", target_bytes="",
                dimensions=dim, dtype=str(dtype),
                expected_leaf_count=math.ceil(n_database / cap)))

print("ImageNet CLIP:", n_database, "database vectors,", len(queries), "queries")
print("Method:", STRATEGY, "indexes:", len(capacities))
print("Results:", OUTPUT)

for levels, capacity, source, target_bytes in capacities:
    index_id = f"imagenet_{STRATEGY}_L{levels}_C{capacity}"
    index_path = INDEXES / f"{index_id}.zarr"
    done = COMPLETED / f"{index_id}.done"
    if done.is_file():
        if index_path.is_dir():
            shutil.rmtree(index_path)
            add_row("index_lifecycle.csv", dict(index_id=index_id,
                time=stamp(), event="index_deleted_after_resume", path=str(index_path)))
        print("Done:", index_id)
        continue
    completed_builds = {r["index_id"]: r for r in read_rows("index_builds.csv")}
    try:
        if index_id in completed_builds:
            if not index_path.is_dir():
                raise RuntimeError("Unfinished index directory missing: " + str(index_path))
            leaf_count = int(completed_builds[index_id]["actual_nonempty_leaves"])
            print("Resuming", index_id)
        else:
            if index_path.exists():
                raise RuntimeError("Unfinished index exists without build CSV: " + str(index_path))
            print("Building", index_id, flush=True)
            prep_id = ""
            prep_peak = 0

            builder = Builder(index_path=str(index_path), levels=levels,
                metric=Metric.IP, is_normalized=True)
            with Memory(index_id, "selection") as m_select:
                begin = perf_counter()
                builder.select_representatives(
                    embeddings_file=str(DATA), target_cluster_items=capacity,
                    strategy=STRATEGY, fallback_batch_rows=BATCH_ROWS,
                    grp_name="train")
                selection_s = perf_counter() - begin
            with Memory(index_id, "build") as m_build:
                begin = perf_counter()
                builder.build(embeddings_file=str(DATA),
                    fallback_batch_rows=BATCH_ROWS, grp_name="train")
                build_s = perf_counter() - begin
            del builder
            index_size = folder_size(index_path) / 2**20
            leaf_count = inspect_index(index_path, index_id, levels, capacity, n_database)
            add_row("index_builds.csv", dict(index_id=index_id, dataset="imagenet",
                strategy=STRATEGY, levels=levels, capacity_vectors=capacity,
                capacity_source=source, representatives=math.ceil(n_database / capacity),
                selection_s=selection_s, build_s=build_s,
                preparation_id=prep_id,
                peak_selection_rss_mib=m_select.peak,
                peak_build_rss_mib=m_build.peak,
                index_size_mib=index_size, actual_nonempty_leaves=leaf_count,
                dataset_vectors=n_database, ecpfs_version=ecp_version,
                started_at=stamp(), status="completed"))
        test_index(index_path, index_id, leaf_count, queries, neighbors)
        done.write_text(stamp(), encoding="utf-8")
        add_row("index_lifecycle.csv", dict(index_id=index_id,
            time=stamp(), event="completed", path=str(index_path)))
        shutil.rmtree(index_path)
        add_row("index_lifecycle.csv", dict(index_id=index_id,
            time=stamp(), event="index_deleted", path=str(index_path)))
        print("Saved results and deleted index:", index_id, flush=True)
    except Exception as error:
        add_row("events.csv", dict(index_id=index_id, time=stamp(),
            stage="build_or_search", message=str(error)))
        print(traceback.format_exc(), flush=True)
        print("Index kept for checking:", index_path)
        raise

print("All", STRATEGY, "ImageNet CLIP experiments finished.")
