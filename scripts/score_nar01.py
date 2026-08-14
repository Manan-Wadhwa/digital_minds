#!/usr/bin/env python3
"""Score v2/NAR01 against its pre-commitments, from the committed JSON alone.

Stdlib only, and every number re-derived from `results.rows` rather than read
out of `results.summary`. Both properties are load-bearing in this repo:

  * `rescore_manipulation.py` was advertised as "score it yourself" and then
    crashed on the machine it was advertised for, because it imported a module
    that dragged in torch (REVIEW.md C9). Nothing here imports anything that
    is not in the standard library.
  * Three pre-registered criteria in this programme have PASSED ON THE WRONG
    PROPERTY. A scorer that reads the summary block written by the same run it
    is scoring cannot catch that; one that recomputes from per-condition rows
    can. `summarise()` in run.py and this file are deliberately two
    implementations of the same arithmetic.

Usage:  python3 scripts/score_nar01.py experiments/v2/NAR01_narration_recipe/results/*.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

CONTINGENCY_BAR = 0.5      # calibration.manipulation.NARRATION_CONTINGENCY_BAR
INVARIANCE_BAND = 0.15     # E17 pre-commitment (1), inherited verbatim
ARMS = ["control", "pool1", "enum", "enum_bal"]


def load(path):
    d = json.loads(Path(path).read_text())
    return d.get("manifest", {}), d["results"]["rows"], d["results"]


def cell(rows, arm, kind):
    return [r for r in rows if r["arm"] == arm and r["kind"] == kind]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3, plus=False):
    if x is None:
        return "  --  "
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def paired(rows, arm_a, arm_b, kind="ORG-B"):
    a = {r["seed"]: r["contingency"] for r in cell(rows, arm_a, kind)
         if r["contingency"] is not None}
    b = {r["seed"]: r["contingency"] for r in cell(rows, arm_b, kind)
         if r["contingency"] is not None}
    common = sorted(set(a) & set(b))
    if not common:
        return None, 0
    return mean([a[s] - b[s] for s in common]), len(common)


def main(paths):
    if not paths:
        print(__doc__)
        return 2
    rows, manifests = [], []
    for p in paths:
        m, r, _res = load(p)
        manifests.append((p, m))
        rows.extend(r)

    print("=" * 78)
    print("v2/NAR01 -- is a narration-only organism installable?")
    print("=" * 78)
    for p, m in manifests:
        print(f"  {Path(p).name}  git {m.get('git_sha','?')}  "
              f"{m.get('model_id','?')}  seeds {m.get('seeds')}")
    seeds = sorted({r["seed"] for r in rows})
    print(f"  {len(rows)} rows, {len(seeds)} seeds: {seeds}")

    # ---------- ORG-B, the organism the axis rests on ----------
    print("\nORG-B by arm")
    print(f"  {'arm':<10} {'n':>2} {'mean cont':>10} {'>bar':>5} "
          f"{'inv':>4} {'mean ratio':>11} {'states':>7} {'ex':>6}")
    for arm in ARMS:
        sub = cell(rows, arm, "ORG-B")
        if not sub:
            continue
        cont = [r["contingency"] for r in sub if r["contingency"] is not None]
        n_bar = sum(1 for c in cont if c > CONTINGENCY_BAR)
        n_inv = sum(1 for r in sub
                    if abs(r["ratio"] - 1.0) <= INVARIANCE_BAND)
        st = sorted({r["n_train_states"] for r in sub})
        ex = sorted({r["n_train_examples"] for r in sub if r["n_train_examples"]})
        print(f"  {arm:<10} {len(sub):>2} {fmt(mean(cont), plus=True):>10} "
              f"{n_bar:>2}/{len(sub):<2} {n_inv:>1}/{len(sub):<2} "
              f"{fmt(mean([r['ratio'] for r in sub])):>11} "
              f"{str(st[0] if len(st)==1 else st):>7} "
              f"{str(ex[0] if len(ex)==1 else ex):>6}")

    # ---------- pre-commitment (1): the load-bearing one ----------
    print("\n(1) POLICY INVARIANCE -- |ratio-1| <= 0.15 for ORG-B  [LOAD-BEARING]")
    verdicts = {}
    for arm in ARMS:
        sub = cell(rows, arm, "ORG-B")
        if not sub:
            continue
        bad = [(r["seed"], round(r["ratio"], 3)) for r in sub
               if abs(r["ratio"] - 1.0) > INVARIANCE_BAND]
        verdicts[arm] = not bad
        status = "OK" if not bad else f"REJECTED -- moved on {len(bad)}: {bad}"
        print(f"  {arm:<10} {status}")
    print("  An arm that fails here is rejected whatever its contingency:")
    print("  a contingent ORG-B whose policy moved is a second function organism.")

    # ---------- pre-commitment (2): the question ----------
    print("\n(2) THE QUESTION -- enum beats control, non-inferior to pool1")
    for a, b in [("enum", "control"), ("enum_bal", "control"),
                 ("pool1", "control"), ("enum", "pool1"), ("enum_bal", "pool1")]:
        d, n = paired(rows, a, b)
        print(f"  {a:<9} - {b:<9}  paired mean {fmt(d, plus=True)}  (n={n})")
    d_ec, _ = paired(rows, "enum", "control")
    d_ep, _ = paired(rows, "enum", "pool1")
    beats_control = d_ec is not None and d_ec > 0
    noninferior = d_ep is not None and d_ep >= 0
    print(f"  enum beats control: {beats_control};  "
          f"non-inferior to pool1: {noninferior}")

    # ---------- pre-commitment (3): the recorded confound ----------
    print("\n(3) THE MATCHED-EXAMPLES CONFOUND (recorded, not controlled)")
    for arm in ARMS:
        sub = cell(rows, arm, "ORG-B")
        if not sub:
            continue
        st = sorted({r["n_train_states"] for r in sub})
        ex = sorted({r["n_train_examples"] for r in sub if r["n_train_examples"]})
        print(f"  {arm:<10} distinct grids {st}   examples {ex}")
    print("  Examples are matched across arms; states are not. If enum loses,")
    print("  state diversity is the first suspect, not the lever.")

    # ---------- pre-commitment (4): the canaries ----------
    print("\n(4) CANARIES")
    d_ratios = {arm: [r["ratio"] for r in cell(rows, arm, "ORG-D")]
                for arm in ARMS}
    d_ratios = {k: v for k, v in d_ratios.items() if v}
    same = len(set(tuple(v) for v in d_ratios.values())) <= 1
    print(f"  ORG-D identical across arms: {same}")
    if not same:
        for arm, v in d_ratios.items():
            print(f"    {arm:<10} {[round(x,3) for x in v]}")
        print("    ORG-D is untrained in every arm. Drift means adapter leakage.")
    bp = cell(rows, "control", "ORG-B'")
    if bp:
        cont = [r["contingency"] for r in bp if r["contingency"] is not None]
        print(f"  ORG-B' (affectless twin) mean contingency "
              f"{fmt(mean(cont), plus=True)} -- should sit near zero")

    # ---------- the verdict ----------
    print("\n" + "=" * 78)
    winner = None
    best = None
    for arm in ARMS:
        if not verdicts.get(arm):
            continue
        sub = cell(rows, arm, "ORG-B")
        cont = [r["contingency"] for r in sub if r["contingency"] is not None]
        m = mean(cont)
        if m is not None and (best is None or m > best):
            best, winner = m, arm
    n_installed = 0
    if winner:
        sub = cell(rows, winner, "ORG-B")
        n_installed = sum(
            1 for r in sub
            if r["contingency"] is not None and r["contingency"] > CONTINGENCY_BAR
            and abs(r["ratio"] - 1.0) <= INVARIANCE_BAND)
    print(f"VERDICT: best policy-invariant arm = {winner}  "
          f"(mean contingency {fmt(best, plus=True)})")
    print(f"         valid ORG-B organisms (contingent AND invariant): "
          f"{n_installed}/{len(cell(rows, winner, 'ORG-B')) if winner else 0}")
    print("  Read the per-arm rows above, never this line alone.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    args = []
    for a in sys.argv[1:]:
        args.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])
    sys.exit(main(args))
