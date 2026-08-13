"""Score v2/ENV02's pre-commitments. Committed with run.py, before any run.

Usage: python3 scripts/score_env02.py <results.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path):
    d = json.loads(Path(path).read_text())
    rows = d.get("results", d)["rows"]
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["kind"], r["seed"]): r for r in rows}
    print("=" * 78)
    print(f"ENV02  {path}   git {d.get('manifest', {}).get('git_sha', '?')}")

    print(f"\n  {'kind':<6} " + " ".join(f"s{s}:share/base" for s in seeds))
    for kind in ("W-D", "W-A", "W-B", "W-B'"):
        cells = [by.get((kind, s)) for s in seeds]
        print(f"  {kind:<6} " + " ".join(
            f"{c['share_vs_base']:.2f}" if c and c.get("share_vs_base")
            else "  - " for c in cells))

    # (2) behaviour gate
    wa = [by[("W-A", s)]["share_vs_base"] for s in seeds
          if by.get(("W-A", s)) and by[("W-A", s)].get("share_vs_base")]
    n_cut = sum(1 for v in wa if v < 0.5)
    inv = all(abs(by[(k, s)]["share_vs_base"] - 1.0) <= 0.15
              for k in ("W-B", "W-B'") for s in seeds
              if by.get((k, s)) and by[(k, s)].get("share_vs_base"))
    ok2 = n_cut >= 4 and inv
    print(f"\n(2) {'PASS' if ok2 else 'FAIL'}  W-A cuts penalised share <0.5x "
          f"base on {n_cut}/{len(seeds)} seeds (needs >=4); B/B' policy "
          f"invariant: {inv}")
    if not ok2:
        print("     -> instruments NOT interpreted; this is a failed build "
              "report, as pre-committed.")
        return

    # (3) the replication question
    print("\n(3) instrument contrasts, W-A vs pooled {W-B, W-B'} "
          "(paired per seed, mean diff):")
    for key in ("I2_self_report", "I4_one_word", "I6_placebo"):
        diffs = []
        for s in seeds:
            b = (by[("W-B", s)][key] + by[("W-B'", s)][key]) / 2
            diffs.append(by[("W-A", s)][key] - b)
        m = sum(diffs) / len(diffs)
        neg = sum(1 for x in diffs if x < 0)
        print(f"     {key:<16} mean {m:+.4f}  signs {neg}-/{len(diffs)-neg}+"
              + ("   (pre-registered direction: negative)"
                 if key != "I6_placebo" else "   (placebo: near zero)"))

    # (4) narration axis under corrected criteria
    print("\n(4) W-B contingency per seed (corrected criteria from day one):")
    for s in seeds:
        r = by[("W-B", s)]
        print(f"     s{s}: presence {r['narration']}  contingency "
              f"{r['contingency']}  measured_narration {r['measured_narration']}")
    print("\n(6) W-A final entropy per seed:",
          [by[("W-A", s)].get("final_entropy") for s in seeds])


if __name__ == "__main__":
    main(sys.argv[1])
