#!/usr/bin/env python3
"""Merge v2/NAR01 shard results into one canonical JSON.

Same contract as `scripts/e16_parallel.py --merge`, which E16 v2 verified: per
cell `set_all_seeds` makes a seed-sharded run bit-identical to a sequential one,
so the merged `rows` are the rows a single-process run would have produced. The
merge recomputes `summary` over the union and DISCARDS each shard's own summary,
because a per-shard summary is computed over four seeds and the pre-commitments
are defined over eight.

Runs on the GPU box (it imports run.py, which imports torch). The merged JSON is
then scored anywhere by `scripts/score_nar01.py`, which is stdlib only.

Usage:
    python3 scripts/nar01_merge.py [--shard-root DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
EXP = _ROOT / "experiments" / "v2" / "NAR01_narration_recipe"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("nar01", EXP / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge(shard_root, out_dir=None, log=print):
    mod = _load_run_module()
    # `shard_*_smoke` directories are gate artefacts, not data. Excluding them
    # by name rather than by mtime, so a re-run cannot quietly fold a smoke
    # result into the science.
    shard_files = sorted(
        p for p in Path(shard_root).glob("shard_*/2*.json")
        if not p.parent.name.endswith("_smoke"))
    if not shard_files:
        raise SystemExit(f"no shard results under {shard_root}")
    shards = [json.loads(p.read_text()) for p in shard_files]

    arms = list(mod.CONFIG["arms"])
    kinds = list(mod.CONFIG["kinds"])
    rows = [r for s in shards for r in s["results"]["rows"]]
    rows.sort(key=lambda r: (r["seed"], arms.index(r["arm"]),
                             kinds.index(r["kind"])))

    seeds = sorted({r["seed"] for r in rows})
    expected = len(seeds) * len(arms) * len(kinds)
    if len(rows) != expected:
        raise SystemExit(
            f"shards hold {len(rows)} rows, expected "
            f"{len(seeds)}x{len(arms)}x{len(kinds)}={expected}")
    dupes = len(rows) - len({(r["seed"], r["arm"], r["kind"]) for r in rows})
    if dupes:
        raise SystemExit(f"{dupes} duplicate (seed, arm, kind) cells; "
                         "shards overlap -- do not merge")

    shas = {s["results"].get("git_sha") or s["manifest"].get("git_sha")
            for s in shards}
    if len(shas) > 1:
        raise SystemExit(f"shards ran different code: {shas}")

    cfg = {**mod.CONFIG, "seeds": seeds}
    results = {"rows": rows, "summary": mod.summarise(rows, cfg)}
    results["elapsed_minutes"] = round(
        sum(s["results"].get("elapsed_minutes", 0) for s in shards), 2)
    results["wall_clock_note"] = (
        "elapsed_minutes is the SUM over shards (GPU work). Shards ran "
        "concurrently on separate boxes, so wall clock is the max, not this.")
    results["git_sha"] = next(iter(shas))
    results["parallel_shards"] = [
        {"file": p.name,
         "seeds": sorted({r["seed"] for r in s["results"]["rows"]}),
         "elapsed_minutes": s["results"].get("elapsed_minutes")}
        for p, s in zip(shard_files, shards)]

    from calibration.runner import RunManifest, save_results
    manifest = RunManifest(
        experiment="v2_NAR01_narration_recipe",
        config={**shards[0]["manifest"]["config"], "seeds": seeds},
        seeds=seeds, model_id=cfg["model_id"],
        device=shards[0]["manifest"].get("device", "cuda"),
        notes=("PARALLEL run: seeds partitioned across two GPU boxes, merged "
               "by scripts/nar01_merge.py. Cells are RNG-independent (per-cell "
               "set_all_seeds), so rows match a sequential run. Partition: "
               f"{[s['seeds'] for s in results['parallel_shards']]}"),
    )
    out_dir = Path(out_dir or EXP / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ad_dir = out_dir / "adapters"
    ad_dir.mkdir(exist_ok=True)
    n_ad = 0
    for shard in sorted(Path(shard_root).glob("shard_*")):
        if shard.name.endswith("_smoke"):
            continue
        for pt in (shard / "adapters").glob("*.pt"):
            (ad_dir / pt.name).write_bytes(pt.read_bytes())
            n_ad += 1
    path = save_results(out_dir, manifest.finish(), results)
    log(f"MERGED {len(rows)} rows ({len(seeds)} seeds x {len(arms)} arms x "
        f"{len(kinds)} kinds) from {len(shards)} shards, {n_ad} adapters "
        f"-> {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", default=str(EXP / "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    merge(a.shard_root, a.out)
