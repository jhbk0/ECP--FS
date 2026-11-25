#!/usr/bin/env python3
"""
ecp_run2.py - build an eCP-FS index on a VIBE-style HDF5 dataset and measure EVERYTHING we can.
No VIBE and no container needed. Same settings as our VIBE adapter:
  metric 'normalized' -> Metric.IP, is_normalized=True ; 'euclidean' -> Metric.L2
  search: new_search(k, search_exp, max_increments=-1, exclude_vec=[]) then end_search

Needs in the HDF5 file: train (N x d), test (Q x d), neighbors (Q x >=k), attribute 'distance'.

Per search_exp one JSON line is appended to --out (a re-run skips finished ones). It keeps every field of
ecp_run.py (so analyze.py still works) and ADDS: recall@1/10/100, per-query latency percentiles, a REAL cold pass
(index files evicted from the OS page cache first, residency checked with fincore when available), CPU user/sys time,
page faults, bytes read (syscalls and from storage), peak memory, machine noise checks (load average and a memory-
bandwidth calibration before each setting), determinism check. Side files next to --out:
  <out>.meta_<tag>.json            machine, versions, dataset, arguments, build breakdown, index size and file count
  <out>.leaf_sizes_<tag>.json      cluster statistics (as before)
  <out>.perquery_<tag>_se<N>.npz   per-query latencies, per-query recall@100 and the returned ids
"""
import argparse, json, os, platform, random, resource, shutil, socket, subprocess, sys, time
import h5py
import numpy as np
from ecpfs import Builder, EmbeddingDtype, Index, Metric


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dir_stats(path):
    total, files = 0, 0
    for root, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(root, f))
                files += 1
            except OSError:
                pass
    return total / 1e9, files


def proc_status():
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                out[k] = int(v.split()[0]) / 1e6      # kB -> GB
    except Exception:
        pass
    return out


def proc_io():
    out = {}
    try:
        for line in open("/proc/self/io"):
            k, v = line.split(":")
            out[k.strip()] = int(v)
    except Exception:
        pass
    return out


def ru():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return dict(user=r.ru_utime, sys=r.ru_stime, minflt=r.ru_minflt, majflt=r.ru_majflt,
                inblock=r.ru_inblock, vcsw=r.ru_nvcsw, ivcsw=r.ru_nivcsw)


def loadavg():
    try:
        return [float(x) for x in open("/proc/loadavg").read().split()[:3]]
    except Exception:
        return None


_CAL = None
def calib_gbps():
    """Single-thread memory-read speed right now (numpy matrix-vector over 134 MB). Noisy neighbours lower it."""
    global _CAL
    if _CAL is None:
        _CAL = (np.ones((32768, 1024), dtype=np.float32), np.ones(1024, dtype=np.float32))
    a, q = _CAL
    a @ q
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); a @ q; best = min(best, time.perf_counter() - t)
    return a.nbytes / 1e9 / best


def evict_page_cache(path):
    """Ask the OS to drop this index's files from the page cache (works without root for files we own)."""
    try:
        os.sync()
    except Exception:
        pass
    n = 0
    for root, _, fs in os.walk(path):
        for f in fs:
            try:
                fd = os.open(os.path.join(root, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
    return n


def resident_fraction(path, sample=300):
    """Share of a random sample of index files that is in the page cache (uses fincore if installed), else None."""
    try:
        files = []
        for root, _, fs in os.walk(path):
            for f in fs:
                if f != "zarr.json":
                    files.append(os.path.join(root, f))
        if not files:
            return None
        random.Random(1).shuffle(files)
        files = files[:sample]
        r = subprocess.run(["fincore", "-b", "-n", "-o", "RES,SIZE"] + files, capture_output=True, text=True, timeout=60)
        res = size = 0
        for line in r.stdout.splitlines():
            p = line.split()
            if len(p) >= 2:
                res += int(p[0]); size += int(p[1])
        return res / size if size else None
    except Exception:
        return None


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return None


def mem_total_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1e6
    except Exception:
        return None


def pct(a, p):
    return float(np.percentile(a, p))


def index_stats(index_path, levels, n_expected):
    """Count the real clusters by reading the shape of every node's id array (no vectors are read).
    Layout: <index>/lvl_1 .. lvl_<levels>; lvl_1 is the root level, the LAST level holds the leaf clusters (item_ids)."""
    out = {"levels": {}}
    leaf_sizes = None
    for lv in range(1, levels + 1):
        ldir = os.path.join(index_path, f"lvl_{lv}")
        nodes = sorted(d for d in os.listdir(ldir) if d.startswith("node_"))
        key = "item_ids" if lv == levels else "node_ids"
        sizes = []
        for nd in nodes:
            zj = os.path.join(ldir, nd, key, "zarr.json")
            try:
                sizes.append(int(json.load(open(zj))["shape"][0]))
            except Exception:
                sizes.append(0)
        a = np.array(sizes)
        info = {"nodes": len(nodes), "min": int(a.min()), "mean": float(a.mean()),
                "max": int(a.max()), "empty": int((a == 0).sum()),
                "what": "items per leaf cluster" if lv == levels else "children per node"}
        if lv == levels:
            leaf_sizes = a
            info.update({"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                         "p99": float(np.percentile(a, 99)), "total_items": int(a.sum()),
                         "expected_items": int(n_expected)})
        out["levels"][f"lvl_{lv}"] = info
    out["n_leaf_clusters"] = out["levels"][f"lvl_{levels}"]["nodes"]
    return out, leaf_sizes


def brute_force_neighbors(train, queries, k, metric, chunk=50_000):
    """Exact top-k for a SUBSET run (the file's neighbors only fit the full train set)."""
    best_s = np.full((len(queries), k), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(queries), k), dtype=np.int64)
    for start in range(0, len(train), chunk):
        X = np.asarray(train[start:start + chunk], dtype=np.float32)
        if metric == "normalized":
            S = queries @ X.T
        else:  # euclidean: larger score = closer
            S = -(np.sum(X * X, axis=1)[None, :] - 2 * queries @ X.T)
        ids = np.arange(start, start + len(X))[None, :].repeat(len(queries), 0)
        allS = np.concatenate([best_s, S], axis=1)
        allI = np.concatenate([best_i, ids], axis=1)
        top = np.argpartition(-allS, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(allS, top, 1)
        best_i = np.take_along_axis(allI, top, 1)
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_i, order, 1)


def run_pass(index, Q, k, exp, max_inc):
    """One pass over all queries. Returns (found ids, latencies in ms, dict of resource deltas)."""
    nq = len(Q)
    lat = np.zeros(nq)
    found = np.zeros((nq, k), dtype=np.int64) - 1
    scores_ok = []
    r0, io0, t0 = ru(), proc_io(), time.perf_counter()
    for i in range(nq):
        s = time.perf_counter()
        items, qid = index.new_search(query=Q[i], k=k, search_exp=exp, max_increments=max_inc, exclude_vec=[])
        index.end_search(qid)
        lat[i] = (time.perf_counter() - s) * 1000.0
        ids = [it_id for _s, it_id in items][:k]
        found[i, :len(ids)] = ids
        if i < 20 and len(items) > 1:
            sc = [x[0] for x in items]
            scores_ok.append("asc" if all(a <= b for a, b in zip(sc, sc[1:])) else
                             "desc" if all(a >= b for a, b in zip(sc, sc[1:])) else "unsorted")
    wall = time.perf_counter() - t0
    r1, io1 = ru(), proc_io()
    d = {key: r1[key] - r0[key] for key in r0}
    d["wall_s"] = wall
    d["qps"] = nq / wall
    d["rchar_per_query"] = (io1.get("rchar", 0) - io0.get("rchar", 0)) / nq
    d["storage_read_per_query"] = (io1.get("read_bytes", 0) - io0.get("read_bytes", 0)) / nq
    d["syscr_per_query"] = (io1.get("syscr", 0) - io0.get("syscr", 0)) / nq
    d["lat_ms"] = dict(mean=float(lat.mean()), p50=pct(lat, 50), p90=pct(lat, 90), p95=pct(lat, 95),
                       p99=pct(lat, 99), max=float(lat.max()))
    return found, lat, d, (scores_ok[0] if scores_ok and len(set(scores_ok)) == 1 else ("mixed" if scores_ok else None))


def recall_at(found, truth, kk):
    kk = min(kk, found.shape[1], truth.shape[1])
    hits = np.array([len(set(found[i, :kk].tolist()) & set(truth[i, :kk].tolist())) for i in range(len(found))])
    return hits / kk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workdir", required=True, help="fast local disk (e.g. /scratch/$SLURM_JOB_ID)")
    ap.add_argument("--out", required=True, help="results file (.jsonl), put it in /home")
    ap.add_argument("--target", type=int, required=True, help="target_cluster_items")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--search-exp", default="1,2,4,8,16,32,64,128,256,512,1024")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--passes", type=int, default=3, help="pass 1 is COLD when --cold is on, later passes are warm")
    ap.add_argument("--cold", type=int, default=1, help="1 = evict the index from the page cache before each search_exp and reopen it")
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N vectors (test runs)")
    ap.add_argument("--max-increments", type=int, default=-1)
    ap.add_argument("--node-chunk-kb", type=int, default=512)
    ap.add_argument("--embedding-dtype", default="", help="'' = keep source dtype, or f32 / f16")
    ap.add_argument("--index-memory-gb", type=float, default=0, help="Index(memory_limit_bytes=...), 0 = library default")
    ap.add_argument("--index-memory-frac", type=float, default=0, help="index memory limit as a FRACTION of the index size on disk (e.g. 0.25); overrides --index-memory-gb")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--deadline-seconds", type=float, default=0, help="do not START a new search_exp after this many seconds")
    args = ap.parse_args()

    t_start = time.time()
    exps = [int(x) for x in args.search_exp.split(",")]
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    try:
        from importlib.metadata import version as _v
        ecp_version = _v("ecpfs")
    except Exception:
        ecp_version = "?"
    log("host", socket.gethostname(), platform.machine(), "python", sys.version.split()[0], "ecpfs", ecp_version)

    dtype_arg = {"": None, "f32": EmbeddingDtype.F32, "f16": EmbeddingDtype.F16}[args.embedding_dtype.lower()]

    with h5py.File(args.data, "r") as f:
        ds_keys = {k: (list(f[k].shape), str(f[k].dtype)) for k in f.keys()}
        ds_attrs = {k: (v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)) for k, v in f.attrs.items()}
        log("keys:", ds_keys, "attrs:", ds_attrs)
        for need in ("train", "test", "neighbors"):
            if need not in f:
                sys.exit(f"STOP: dataset has no '{need}' (has {list(f.keys())})")
        metric_name = ds_attrs.get("distance")
        if metric_name == "normalized":
            metric, is_norm = Metric.IP, True
        elif metric_name == "euclidean":
            metric, is_norm = Metric.L2, False
        else:
            sys.exit(f"STOP: metric attribute is {metric_name!r}; this script only handles 'normalized' and 'euclidean'")

        train = f["train"]
        N_full, d = train.shape
        N = min(N_full, args.max_train) if args.max_train else N_full
        Q = np.asarray(f["test"][: args.n_queries], dtype=np.float32)
        nq = len(Q)
        log(f"dataset: {N_full} x {d}; using N={N}, {nq} queries, k={args.k}, metric={metric_name}")

        truth = None
        if args.build_only:
            log("build-only mode: skipping ground truth")
        elif N < N_full:
            log("subset run: computing exact neighbours by brute force")
            truth = brute_force_neighbors(train[:N], Q, args.k, metric_name)
        else:
            truth = np.asarray(f["neighbors"][:nq, : args.k])
            for q in range(min(5, nq)):
                a = np.asarray(train[int(truth[q, 0])], dtype=np.float32)
                b = np.asarray(train[int(truth[q, -1])], dtype=np.float32)
                sa, sb = float(Q[q] @ a), float(Q[q] @ b)
                if metric_name == "normalized" and sa + 1e-4 < sb:
                    sys.exit(f"STOP: neighbors look wrong for query {q}: first score {sa} < last score {sb}")
        if truth is not None:
            log("ground truth ready", truth.shape)

        tag = f"t{args.target}_l{args.levels}_n{N}_c{args.node_chunk_kb}" + (f"_{args.embedding_dtype.lower()}" if args.embedding_dtype else "")
        index_path = os.path.join(args.workdir, f"index_{tag}.zarr")
        ready = index_path + ".READY"
        build_info = {}
        if os.path.exists(ready):
            log("index already built, reusing", index_path)
            try:
                build_info = json.load(open(ready))
            except Exception:
                build_info = {}
        else:
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            emb = os.path.join(args.workdir, f"emb_{tag}.h5")
            log("copying vectors to", emb, "(in chunks, low memory)")
            t0 = time.time()
            with h5py.File(emb, "w") as g:
                dset = g.create_dataset("embeddings", shape=(N, d), dtype="float32", chunks=(min(N, 10_000), d))
                step = 100_000
                for s in range(0, N, step):
                    dset[s:s + step] = np.asarray(train[s:min(s + step, N)], dtype=np.float32)
            copy_s = time.time() - t0
            log(f"copy done in {copy_s:.0f}s, file {os.path.getsize(emb) / 1e9:.1f} GB")
            b = Builder(index_path=index_path, levels=args.levels, metric=metric, is_normalized=is_norm,
                        embedding_dtype=dtype_arg, node_chunk_bytes=args.node_chunk_kb * 1024)
            t0 = time.time()
            log("select_representatives ...")
            b.select_representatives(embeddings_file=emb, target_cluster_items=args.target,
                                     strategy="offset", fallback_batch_rows=100_000)
            select_s = time.time() - t0
            t1 = time.time()
            log(f"select done in {select_s:.0f}s; build ...")
            b.build(embeddings_file=emb, fallback_batch_rows=100_000)
            build_s = time.time() - t1
            os.remove(emb)
            gb, nfiles = dir_stats(index_path)
            build_info = dict(copy_s=copy_s, select_s=select_s, build_only_s=build_s, build_total_s=select_s + build_s,
                              peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                              index_gb=gb, index_files=nfiles)
            json.dump(build_info, open(ready, "w"))
            log(f"BUILD DONE in {select_s + build_s:.0f}s (select {select_s:.0f}s + build {build_s:.0f}s); "
                f"index {gb:.1f} GB in {nfiles} files; build-process peak RSS {build_info['peak_rss_gb']:.1f} GB")

    stats, leaf_sizes = index_stats(index_path, args.levels, N)
    log("CLUSTERS CREATED:", stats["n_leaf_clusters"], "leaf clusters (target was ~", N // args.target, ")")
    for lv, inf in stats["levels"].items():
        log(f"  {lv}: {inf['nodes']} nodes, {inf['what']}: min {inf['min']}, mean {inf['mean']:.1f}, max {inf['max']}, empty {inf['empty']}")
    leaf = stats["levels"][f"lvl_{args.levels}"]
    log(f"  leaf clusters: median {leaf['median']:.0f}, p95 {leaf['p95']:.0f}, p99 {leaf['p99']:.0f}; "
        f"items in leaves {leaf['total_items']} of {leaf['expected_items']}")
    if leaf["total_items"] != leaf["expected_items"]:
        log("WARNING: leaf item count differs from number of vectors - check the cluster counts before trusting them")
    sizes_file = os.path.splitext(args.out)[0] + f".leaf_sizes_{tag}.json"
    json.dump({"tag": tag, "stats": stats, "leaf_sizes": leaf_sizes.tolist()}, open(sizes_file, "w"))

    meta = dict(tag=tag, host=socket.gethostname(), cpu=cpu_model(), cpus_visible=len(os.sched_getaffinity(0)),
                pinned_cpus=sorted(os.sched_getaffinity(0))[:8], mem_total_gb=mem_total_gb(), kernel=platform.release(),
                python=sys.version.split()[0], numpy=np.__version__, ecpfs=ecp_version, ecp_sha=os.environ.get("ECP_SHA"),
                slurm={k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST")},
                dataset=dict(file=os.path.basename(args.data), keys=ds_keys, attrs=ds_attrs, n_used=N, d=d),
                args=vars(args), build=build_info, index_gb=dir_stats(index_path)[0], index_files=dir_stats(index_path)[1],
                loadavg_start=loadavg(), calib_gbps_start=calib_gbps(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                note="calib_gbps = single-thread memory read speed; a low value during a run means a noisy node")
    json.dump(meta, open(os.path.splitext(args.out)[0] + f".meta_{tag}.json", "w"), indent=1)
    log(f"meta saved; index {meta['index_gb']:.1f} GB in {meta['index_files']} files; memory-bandwidth calibration {meta['calib_gbps_start']:.1f} GB/s")
    if args.build_only:
        log("BUILD ONLY: finished (index kept in", index_path, ")")
        return

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                if r.get("tag") == tag and r.get("k") == args.k and r.get("n_queries") == nq:
                    done.add(r["search_exp"])
            except Exception:
                pass

    if args.index_memory_frac:
        args.index_memory_gb = args.index_memory_frac * dir_stats(index_path)[0]
        log(f"index memory limit = {args.index_memory_frac:.2f} x index size = {args.index_memory_gb:.2f} GB")
    mem_kw = {"memory_limit_bytes": int(args.index_memory_gb * 1e9)} if args.index_memory_gb else {}
    index = None
    for exp in exps:
        if exp in done:
            log(f"search_exp={exp}: already in results, skipping")
            continue
        if args.deadline_seconds and time.time() - t_start > args.deadline_seconds:
            log(f"deadline reached, not starting search_exp={exp} or later")
            break
        cal0, load0 = calib_gbps(), loadavg()
        evicted, res_before = None, None
        if args.cold:
            if index is not None:
                index.close(); index = None
            evicted = evict_page_cache(index_path)
            res_before = resident_fraction(index_path)
        try:   # reset the process peak-RSS (VmHWM) so rss_gb_peak belongs to THIS setting only
            open("/proc/self/clear_refs", "w").write("5")
        except Exception:
            pass
        t0 = time.perf_counter()
        if index is None:
            index = Index(index_path, **mem_kw)
        open_s = time.perf_counter() - t0
        rss_open = proc_status().get("VmRSS")

        passes, found_first, lat_all, order = [], None, [], None
        for p in range(args.passes):
            found, lat, dres, order_p = run_pass(index, Q, args.k, exp, args.max_increments)
            order = order or order_p
            if p == 0:
                found_first = found
                res_after_p1 = resident_fraction(index_path) if args.cold else None
            dres["cold"] = bool(args.cold and p == 0)
            passes.append(dres)
            lat_all.append(lat)
        deterministic = None
        if args.passes > 1:
            deterministic = bool(np.array_equal(found_first, found))
        r1 = recall_at(found_first, truth, 1) if order in ("asc", "desc") else None
        r10 = recall_at(found_first, truth, 10) if order in ("asc", "desc") else None
        r100 = recall_at(found_first, truth, args.k)
        qps_list = [x["qps"] for x in passes]
        cal1, load1, st = calib_gbps(), loadavg(), proc_status()
        np.savez_compressed(os.path.splitext(args.out)[0] + f".perquery_{tag}_se{exp}.npz",
                            lat_ms=np.array(lat_all), recall100=r100, found=found_first.astype(np.int32))
        rec = dict(tag=tag, dataset=os.path.basename(args.data), n_train=N, d=d, target=args.target,
                   levels=args.levels, search_exp=exp, k=args.k, n_queries=nq, recall=float(r100.mean()),
                   qps=max(qps_list), qps_passes=qps_list, build_seconds=build_info.get("build_total_s"),
                   n_leaf_clusters=stats["n_leaf_clusters"], leaf_mean=leaf["mean"], leaf_median=leaf["median"],
                   leaf_p99=leaf["p99"], leaf_max=leaf["max"], leaf_empty=leaf["empty"],
                   host=socket.gethostname(), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                   # ---- new in ecp_run2 ----
                   recall_at_1=None if r1 is None else float(r1.mean()), recall_at_10=None if r10 is None else float(r10.mean()),
                   recall_at_k_p05=pct(r100, 5), recall_at_k_p50=pct(r100, 50), recall_at_k_min=float(r100.min()),
                   result_order=order, deterministic=deterministic,
                   qps_cold=qps_list[0] if args.cold else None, qps_warm_best=max(qps_list[1:]) if (args.cold and len(qps_list) > 1) else None,
                   passes=passes, node_chunk_kb=args.node_chunk_kb, embedding_dtype=args.embedding_dtype or "source",
                   index_gb=meta["index_gb"], index_files=meta["index_files"],
                   index_open_s=open_s, rss_gb_after_open=rss_open, rss_gb_now=st.get("VmRSS"), rss_gb_peak=st.get("VmHWM"),
                   evicted_files=evicted, resident_frac_after_evict=res_before,
                   resident_frac_after_pass1=res_after_p1 if args.cold else None,
                   calib_gbps_before=cal0, calib_gbps_after=cal1, loadavg_before=load0, loadavg_after=load1)
        with open(args.out, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        p1 = passes[0]
        log(f"search_exp={exp:5d} recall@{args.k}={r100.mean():.4f} QPS best={max(qps_list):9.1f} "
            f"(pass1{' COLD' if args.cold else ''} {qps_list[0]:.1f}; p50 {p1['lat_ms']['p50']:.1f} ms p99 {p1['lat_ms']['p99']:.1f} ms; "
            f"majflt {p1['majflt']}, storage read/query {p1['storage_read_per_query']/1e6:.1f} MB; "
            f"cpu user {p1['user']:.1f}s sys {p1['sys']:.1f}s; RSS {st.get('VmRSS', 0):.1f} GB; calib {cal0:.1f}->{cal1:.1f} GB/s)")
    if index is not None:
        index.close()
    log("ALL DONE")


if __name__ == "__main__":
    main()
