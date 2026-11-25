# Benchmarking eCP-FS on the ITU HPC

Scripts for the eCP-FS benchmarks (ITU MSc project, supervisor Omar Shahbaz Khan).
Everything runs as Slurm jobs on node `cn8`, one core, one thread, one job at a time.

## standalone/ : eCP-FS alone (own script)
| File | What it does |
|---|---|
| `stage1_install.job` | conda environment `~/ecp_env` with eCP-FS 0.9.101 |
| `stage8_*.job` | download the four VIBE datasets into `~/vibe_data` |
| `ecp_run2.py` | build + search + cold/warm passes for one cluster size |
| `stage10_all.job` | the full run: 7 cluster sizes, budgets 3-800 MB, cache limits; array 0-3 = ccnews, imagenet, laion, msmarco |

Output: `~/ecp_results/all_<dataset>_C<KB>kb*.jsonl` (+ meta, leaf sizes).

## vibe/ : the same index through VIBE's own run.py (`--local`, no containers)
Run in this order:
1. `stage11_install_vibe.job` : `~/vibe_env` (Faiss, hnswlib) + VIBE at commit d2c94d4, applies `vibe_hnswlib.patch` (M = 16, 48)
2. `stage12_install_ecp_in_vibe.job` : adds `vibe_ecpfs.patch` (new algorithm module `ecpfs`)
3. `stage11a_vibe_ivf.job` (Faiss IVF-Flat), `stage12_vibe_ecpfs.job` (eCP-FS), `stage11b_vibe_hnsw.job` (hnswlib, not msmarco); submit with `--nodelist=cn8` and `--dependency=afterany:<jobid>` so they never overlap
4. `stage13_collect.job` + `collect_vibe.py` : recall@100 and speed table from the result files

VIBE: all 1000 queries, k = 100, 5 runs, best run reported.

## analysis/
- `results/` : the raw stand-alone results (192 small files, 4 MB): per-setting `.jsonl`, meta and leaf-size files
- `all_results_summary.csv` : one row per setting (222)
- `an3.py` : tables (run inside `results/`: `python3 ../an3.py`)
- `mk_paper.py` : Figures 2 and 3. `python3 mk_paper.py results ../figures`
- `fig_protocol.py` : Figure 1 (protocol diagram), writes `fig_protocol.pdf` in the current folder
The finished PDFs are in `figures/`. Needs python3, numpy, matplotlib.

## Notes
- Recall: stand-alone = identifier overlap; VIBE results = distance rule (dist <= true 100th distance + 1e-3).
- The datasets (tens of GB) and the VIBE result files are not in this repository; the stand-alone results are.
