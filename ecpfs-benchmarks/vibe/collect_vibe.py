"""Read every VIBE result file under ~/vibe_run_*/results and write ONE csv table.

recall@100 = mean over the queries of  (number of returned neighbours whose distance <= the true 100th distance + 1e-3) / 100
(this is the rule VIBE's own export_results.py uses). Ground truth comes from the 'distances' of the dataset file that
sits next to the results (data/<dataset>.hdf5). Only reads; never changes a result file.
usage: python3 collect_vibe.py OUT.csv RUN_DIR [RUN_DIR ...]
"""
import csv, glob, os, re, sys
import h5py
import numpy as np

out, run_dirs = sys.argv[1], sys.argv[2:]
rows, truth_cache = [], {}
for rd in run_dirs:
    for fn in sorted(glob.glob(os.path.join(rd, "results", "*", "*", "*", "*.hdf5"))):
        parts = fn.split(os.sep)
        dataset, algo = parts[-4], parts[-2]
        key = (rd, dataset)
        if key not in truth_cache:
            with h5py.File(os.path.join(rd, "data", dataset + ".hdf5"), "r") as t:
                truth_cache[key] = t["distances"][:]
        true = truth_cache[key]
        try:
            h = h5py.File(fn, "r")
        except OSError as e:          # a file that is still being written by a running job
            print("skipped (not readable yet):", fn, e)
            continue
        with h:
            for g in h:
                a, d = h[g].attrs, h[g]["distances"][:]
                n = min(len(d), len(true))
                rec = float(np.mean([(d[i][:100] <= true[i][99] + 1e-3).sum() / 100 for i in range(n)]))
                name = str(a.get("name", ""))
                row = dict(algo=algo, dataset=dataset, name=name, group=g, queries=n, recall100=round(rec, 5),
                           qps=float(a.get("best_qps", np.nan)), build_s=float(a.get("build_time", np.nan)),
                           run_count=int(a.get("run_count", -1)), file=os.path.basename(fn))
                m = re.search(r"cluster_kb=(\d+)", name); row["cluster_kb"] = int(m.group(1)) if m else ""
                m = re.search(r"search_exp=(\d+)", name); row["search_exp"] = int(m.group(1)) if m else ""
                row["index_bytes_on_disk"] = int(a["index_bytes_on_disk"]) if "index_bytes_on_disk" in a else ""
                rows.append(row)
cols = ["algo", "dataset", "name", "cluster_kb", "search_exp", "recall100", "qps", "build_s", "index_bytes_on_disk",
        "queries", "run_count", "group", "file"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("wrote", out, "rows:", len(rows))
by = {}
for r in rows:
    by.setdefault((r["algo"], r["dataset"]), []).append(r)
print(f"{'algorithm':16s} {'dataset':32s} {'settings':>8s} {'max recall':>10s} {'queries':>8s} {'runs':>5s}")
for (al, ds), rs in sorted(by.items()):
    print(f"{al:16s} {ds:32s} {len(rs):8d} {max(r['recall100'] for r in rs):10.3f} {min(r['queries'] for r in rs):8d} {min(r['run_count'] for r in rs):5d}")
