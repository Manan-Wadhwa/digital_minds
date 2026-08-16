#!/usr/bin/env python
"""Does ORG-C's aversive remark track the PENALISED tile, or any coloured tile?

E16 v2 measured narration contingency against penalised-tile adjacency only.
The novel-glyph probe swaps the penalised glyph in place, so a remark that
fires on "some non-path tile is adjacent" would pass both. This script
regenerates the 48 audit grids per seed (deterministic from the seed via
E16's own `seed_draws`), verifies them against the stored `nar_adjacent`,
and splits every organism's stored generations into three state classes:

    PEN   penalised tile adjacent (gold may or may not be)
    GOLD  penalised NOT adjacent, rewarded tile adjacent
    NONE  neither adjacent

Reports the aversive-remark rate per class for the trained glyph and the
novel-glyph generations, per kind, pooled and per seed. CPU only.

Usage: .venv/bin/python scripts/gold_adjacency_control.py
"""
from __future__ import annotations

import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibration.manipulation import _says_aversive  # noqa: E402

JSON = ROOT / "experiments/E16_calibrated_loading_map/results/20260814T021231Z_95e6e2b8e2df.json"
spec = importlib.util.spec_from_file_location("e16", ROOT / "experiments/E16_calibrated_loading_map/run.py")
e16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e16)

d = json.loads(JSON.read_text())
rows = d["results"]["rows"]
cfg = d["manifest"]["config"]

classes_by_seed = {}
for seed in sorted({r["seed"] for r in rows}):
    dr = e16.seed_draws(seed, cfg)
    pen, rew = dr["pen"], dr["rew"]
    cls = []
    adj = []
    for g, dests in dr["nar_states"]:
        p = any(t == pen for t in dests)
        gd = any(t == rew for t in dests)
        adj.append(p)
        cls.append("PEN" if p else ("GOLD" if gd else "NONE"))
    # verify against the stored flags of any row for this seed
    stored = next(r["nar_adjacent"] for r in rows if r["seed"] == seed)
    assert stored == adj, f"seed {seed}: regenerated adjacency differs from stored"
    classes_by_seed[seed] = cls

census = {c: sum(cls.count(c) for cls in classes_by_seed.values()) for c in ("PEN", "GOLD", "NONE")}
print("audit-state census over 12 seeds:", census)


def rate(texts, cls, want):
    sel = [t for t, c in zip(texts, cls) if c == want]
    return (sum(_says_aversive(t) for t in sel) / len(sel)) if sel else float("nan")


out = {}
for kind in ["ORG-B", "ORG-B'", "ORG-C"]:
    for field, label in (("generations", "trained glyph"), ("generations_novel", "novel glyph")):
        per_seed = {c: [] for c in ("PEN", "GOLD", "NONE")}
        pooled_num = {c: 0 for c in per_seed}
        pooled_den = {c: 0 for c in per_seed}
        for r in rows:
            if r["kind"] != kind:
                continue
            cls = classes_by_seed[r["seed"]]
            texts = r[field]
            for c in per_seed:
                sel = [t for t, k in zip(texts, cls) if k == c]
                if sel:
                    hits = sum(_says_aversive(t) for t in sel)
                    per_seed[c].append(hits / len(sel))
                    pooled_num[c] += hits
                    pooled_den[c] += len(sel)
        pooled = {c: pooled_num[c] / pooled_den[c] if pooled_den[c] else float("nan") for c in per_seed}
        # per-seed GOLD − NONE and PEN − GOLD contrasts where both defined
        gn = [g - n for g, n in zip(per_seed["GOLD"], per_seed["NONE"])]
        pg = [p - g for p, g in zip(per_seed["PEN"], per_seed["GOLD"])]
        out[(kind, label)] = dict(pooled=pooled, n=pooled_den, gold_minus_none=gn, pen_minus_gold=pg)
        print(f"\n{kind:7s} {label:13s} pooled rate  PEN {pooled['PEN']:.3f} (n={pooled_den['PEN']})  "
              f"GOLD-only {pooled['GOLD']:.3f} (n={pooled_den['GOLD']})  NONE {pooled['NONE']:.3f} (n={pooled_den['NONE']})")
        if gn:
            print(f"        per-seed GOLD−NONE mean {st.mean(gn):+.3f} sd {st.pstdev(gn):.3f} "
                  f"signs +{sum(x>0 for x in gn)}/0{sum(x==0 for x in gn)}/−{sum(x<0 for x in gn)}  (n seeds {len(gn)})")
        if pg:
            print(f"        per-seed PEN−GOLD  mean {st.mean(pg):+.3f} sd {st.pstdev(pg):.3f} "
                  f"signs +{sum(x>0 for x in pg)}/0{sum(x==0 for x in pg)}/−{sum(x<0 for x in pg)}")

Path(ROOT / "writing" / "gold_adjacency_control.json").write_text(
    json.dumps({f"{k[0]}|{k[1]}": v for k, v in out.items()}, indent=1))
print("\nwrote writing/gold_adjacency_control.json")
