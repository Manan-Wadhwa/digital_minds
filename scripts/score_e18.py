"""Score E18's pre-commitments off the per-cell numbers. Committed BEFORE the
run -- after REVIEW.md R1 that ordering is the point, not a nicety.

Usage: python3 scripts/score_e18.py <results.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EMITS_BAR = 0.5
RATIO_SLACK = 0.05
CONTROL_MUST_WRECK = 3   # of 4 seeds -- pre-commitment (2)


def main(path):
    d = json.loads(Path(path).read_text())
    r = d.get("results", d)
    rows = r["rows"]
    coefs = sorted({x["coef"] for x in rows})
    seeds = sorted({x["seed"] for x in rows})
    cell = {(x["coef"], x["seed"]): x for x in rows}

    print("=" * 78)
    print(f"E18  {path}")
    print(f"  git {d.get('manifest', {}).get('git_sha', '?')}   "
          f"{r.get('elapsed_minutes', '?')} min   "
          f"coefs {coefs}   seeds {seeds}")
    print(f"\n  {'coef':>6} | " + " | ".join(f"seed {s}: ratio/emits" for s in seeds))
    for c in coefs:
        cells = [cell[(c, s)] for s in seeds]
        print(f"  {c:>6} | " + " | ".join(
            f"{x['ratio']:.3f}/{x['emits_move']:.2f}" for x in cells))

    ctrl = [cell[(0.0, s)] for s in seeds]
    wrecked = [x for x in ctrl if x["emits_move"] < EMITS_BAR]
    ok2 = len(wrecked) >= CONTROL_MUST_WRECK
    print(f"\n(2) {'PASS' if ok2 else 'FAIL'}  control arm reproduces the "
          f"disease: {len(wrecked)}/{len(seeds)} seeds wrecked "
          f"(needs >= {CONTROL_MUST_WRECK})")
    if not ok2:
        print("     -> NO ARM IS INTERPRETABLE; this run measured a "
              "different world than E16. Stop here.")

    healthy_ctrl = [x["ratio"] for x in ctrl if x["emits_move"] >= EMITS_BAR]
    base = (sum(healthy_ctrl) / len(healthy_ctrl)) if healthy_ctrl else None
    print(f"     control mean ratio over its non-wrecked seeds: "
          f"{base if base is None else round(base, 4)}")

    passing = []
    for c in coefs:
        if c == 0.0:
            continue
        cells = [cell[(c, s)] for s in seeds]
        all_emit = all(x["emits_move"] >= EMITS_BAR for x in cells)
        mean_ratio = sum(x["ratio"] for x in cells) / len(cells)
        ratio_ok = base is None or mean_ratio <= base + RATIO_SLACK
        verdict = all_emit and ratio_ok
        passing += [c] if verdict else []
        print(f"(3) coef {c}: {'PASS' if verdict else 'FAIL'}  "
              f"all-emit={all_emit}  mean ratio {mean_ratio:.4f} "
              f"(bar {'n/a' if base is None else round(base + RATIO_SLACK, 4)})")

    if ok2 and passing:
        print(f"\n(4) CHOSEN COEFFICIENT: {min(passing)} (smallest passing). "
              f"E16 CONFIG rl_move_mass_coef should be set to this value.")
    elif ok2:
        print("\n(4) NO COEFFICIENT PASSES. E16's rl_move_mass_coef reverts "
              "to 0.0 and the merged fix does not work as argued -- report "
              "this with the same prominence as a pass.")
    print("\n(5) entropy columns, for the interaction threat:")
    for c in coefs:
        es = [cell[(c, s)].get("final_policy_entropy") for s in seeds]
        print(f"     coef {c}: final policy entropy per seed {es}")


if __name__ == "__main__":
    main(sys.argv[1])
