"""Seed-parallel driver for the E16 loading map: K worker processes, one merge.

WHY THIS IS SAFE, WHICH IS THE ONLY INTERESTING QUESTION

Parallelism usually costs this repo the thing it values most:
bit-reproducibility. Here it does not, for a reason that is a property of the
experiment's own design: `run.py` calls `set_all_seeds(seed)` at the top of
every seed's prelude AND again before every kind, so each (seed, kind) cell's
RNG state is fully derived from the seed and consumes nothing from any other
cell. Cells are RNG-independent by construction. Partitioning SEEDS across
PROCESSES therefore reproduces the sequential run exactly -- E18 demonstrated
per-cell bit-reproduction across machines and thirteen days.

THREADS would not be safe: `set_all_seeds` writes the process-global CPU and
CUDA generators, and two threads interleaving draws on them would produce
organisms that exist nowhere else. Hence processes, each with its own model
replica (~8 GB bf16 + training activations; budget ~16 GB per worker).

Usage, on the GPU host:

    python scripts/e16_parallel.py --workers 4            # spawn + merge
    python scripts/e16_parallel.py --worker 0,1,2 --out D # one worker (internal)
    python scripts/e16_parallel.py --merge                # merge existing shards

The merged JSON is byte-equivalent in `rows` to a sequential run (same cells,
same values, seed-major order); `analyse` is recomputed over the full row set
at merge time, and each shard's own subset-analyse is discarded. Shard
adapters are collected into the canonical results/adapters/. The manifest
records the worker partition in `notes`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "E16_calibrated_loading_map"
KIND_ORDER = ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"]

_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load_run_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e16_run_par", EXP / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()
    return model, tok


def worker(seed_list, out_dir, log_path):
    mod = _load_run_module()
    cfg = {**mod.CONFIG, "seeds": seed_list}
    model, tok = _load_model(cfg["model_id"])
    mod.run(model, tok, cfg, out_dir=out_dir, log_path=log_path)


def merge(shard_root, out_dir=None, log=print):
    """Combine shard JSONs into one canonical result set."""
    mod = _load_run_module()
    shard_files = sorted(Path(shard_root).glob("shard_*/2*.json"))
    if not shard_files:
        raise SystemExit(f"no shard results under {shard_root}")
    shards = [json.loads(p.read_text()) for p in shard_files]

    rows = [r for s in shards for r in s["results"]["rows"]]
    rows.sort(key=lambda r: (r["seed"], KIND_ORDER.index(r["kind"])))
    seeds = sorted({r["seed"] for r in rows})
    expected = len(seeds) * len(KIND_ORDER)
    if len(rows) != expected:
        raise SystemExit(f"shards hold {len(rows)} rows, expected {expected}")

    cfg = {**mod.CONFIG, "seeds": seeds}
    results = mod.analyse(rows, cfg)
    results["rows"] = rows
    results["placebo_selection"] = shards[0]["results"]["placebo_selection"]
    sel = [s["results"]["placebo_selection"]["p_null"]["pair"] for s in shards]
    if any(p != sel[0] for p in sel):
        raise SystemExit("shards disagree on placebo selection -- the "
                         "untrained-model table was not deterministic; stop")
    results["elapsed_minutes"] = round(
        sum(s["results"].get("elapsed_minutes", 0) for s in shards), 2)
    results["git_sha"] = shards[0]["results"].get("git_sha")
    results["parallel_shards"] = [
        {"file": p.name, "seeds": sorted({r["seed"]
                                          for r in s["results"]["rows"]})}
        for p, s in zip(shard_files, shards)]

    from calibration.runner import RunManifest, save_results
    manifest = RunManifest(
        experiment="E16_calibrated_loading_map",
        config={**shards[0]["manifest"]["config"], "seeds": seeds},
        seeds=seeds,
        model_id=cfg["model_id"],
        device=shards[0]["manifest"].get("device", "cuda"),
        notes=("PARALLEL run: seeds partitioned across processes, merged by "
               "scripts/e16_parallel.py; cells are RNG-independent (per-cell "
               "set_all_seeds), so rows are identical to a sequential run. "
               f"Partition: {[s['seeds'] for s in results['parallel_shards']]}"),
    )
    out_dir = Path(out_dir or EXP / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ad_dir = out_dir / "adapters"
    ad_dir.mkdir(exist_ok=True)
    n_ad = 0
    for shard in sorted(Path(shard_root).glob("shard_*")):
        for pt in (shard / "adapters").glob("*.pt"):
            (ad_dir / pt.name).write_bytes(pt.read_bytes())
            n_ad += 1
    path = save_results(out_dir, manifest.finish(), results)
    log(f"MERGED {len(rows)} rows from {len(shards)} shards, "
        f"{n_ad} adapters -> {path}")
    return path


def spawn(n_workers, shard_root, log_path):
    mod = _load_run_module()
    seeds = list(mod.CONFIG["seeds"])
    parts = [seeds[i::n_workers] for i in range(n_workers)]
    procs = []
    for i, part in enumerate(parts):
        out = Path(shard_root) / f"shard_{i}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__,
               "--worker", ",".join(map(str, part)),
               "--out", str(out), "--log", str(out / "log.txt")]
        env = dict(os.environ)
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(20)   # stagger model loads so peak host RAM stays sane
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker exit codes {codes}; not merging")
    return merge(shard_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int)
    ap.add_argument("--worker", type=str, help="comma-separated seed subset")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", type=str)
    ap.add_argument("--log", type=str)
    ap.add_argument("--shard-root", type=str,
                    default=str(EXP / "results" / "parallel"))
    a = ap.parse_args()
    if a.worker:
        worker([int(s) for s in a.worker.split(",")], a.out, a.log)
    elif a.merge:
        merge(a.shard_root)
    elif a.workers:
        spawn(a.workers, a.shard_root, a.log or None)
    else:
        ap.error("pass --workers N, --worker SEEDS, or --merge")
