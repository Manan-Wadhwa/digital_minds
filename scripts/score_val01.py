"""Score v2/VAL01's pre-commitments. Committed with run.py, before any run.

Usage: python3 scripts/score_val01.py <results.json>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# E16 raw_measured committed |d| analogs for the ECHO comparison bar (2):
# I2 0.4673, I4 (raw_measured) -- the discriminator uses half these.
E16_I2_D, E16_I4_D = 0.4673, 0.9711


def paired(rows, cond, key):
    base = {r["seed"]: r[key] for r in rows if r["condition"] == "P-NONE"}
    d = [r[key] - base[r["seed"]] for r in rows if r["condition"] == cond]
    n = len(d)
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    return m, sd, t, sum(1 for x in d if x < 0), sum(1 for x in d if x > 0)


def main(path):
    d = json.loads(Path(path).read_text())
    rows = d.get("results", d)["rows"]
    seeds = sorted({r["seed"] for r in rows})
    print("=" * 78)
    print(f"VAL01  {path}")
    print(f"  git {d.get('manifest', {}).get('git_sha', '?')}  "
          f"{len(seeds)} seeds")

    print(f"\n  {'condition':<11} {'mean ratio':>10}  (penalised-landing / chance)")
    for cond in ("P-NONE", "P-AVOID", "P-APPROACH", "P-NARRATE"):
        rs = [r["ratio"] for r in rows if r["condition"] == cond]
        print(f"  {cond:<11} {sum(rs)/len(rs):>10.3f}")

    m_av, _, _, neg_av, _ = paired(rows, "P-AVOID", "ratio")
    m_ap, _, _, _, pos_ap = paired(rows, "P-APPROACH", "ratio")
    ok1 = (m_av < 0 and neg_av > len(seeds) // 2
           and m_ap > 0 and pos_ap > len(seeds) // 2)
    print(f"\n(1) {'PASS' if ok1 else 'FAILED CONTROL'}  instruction moves "
          f"behaviour: AVOID paired ratio shift {m_av:+.3f} "
          f"({neg_av}/{len(seeds)} negative), APPROACH {m_ap:+.3f} "
          f"({pos_ap}/{len(seeds)} positive)")
    if not ok1:
        print("     -> every instrument null below is uninterpretable.")

    print("\n(2) the discriminator (carried reads, paired vs P-NONE):")
    shifts = {}
    for cond in ("P-AVOID", "P-APPROACH", "P-NARRATE"):
        for key, label in (("I2_carried", "I2"), ("I4_carried", "I4"),
                           ("I5_carried", "I5"), ("I6a_carried", "I6a")):
            m, sd, t, neg, pos = paired(rows, cond, key)
            shifts[(cond, label)] = (m, sd, t)
            print(f"     {cond:<11} {label:<4} shift {m:+.4f}  sd {sd:.3f}  "
                  f"t {t:+.2f}  signs {neg}-/{pos}+")

    av2, av4 = shifts[("P-AVOID", "I2")][0], shifts[("P-AVOID", "I4")][0]
    ap2, ap4 = shifts[("P-APPROACH", "I2")][0], shifts[("P-APPROACH", "I4")][0]
    se2 = shifts[("P-AVOID", "I2")][1] / math.sqrt(len(seeds))
    se4 = shifts[("P-AVOID", "I4")][1] / math.sqrt(len(seeds))
    echo = (abs(av2) >= E16_I2_D / 2 or abs(av4) >= E16_I4_D / 2) and \
           (av2 * ap2 < 0 or av4 * ap4 < 0)
    noecho = ok1 and abs(av2) <= 2 * se2 and abs(av4) <= 2 * se4
    verdict = "ECHO" if echo else ("NO-ECHO" if noecho else "PARTIAL")
    print(f"\n     VERDICT (pre-committed classes): {verdict}")
    print("     ECHO -> visible policy alone can drive the instruments; the "
          "tautology objection gains force.")
    print("     NO-ECHO -> weight installation carries something instruction "
          "does not; the state reading strengthens.")

    print("\n(3) bare reads (must be ~condition-independent; deviation = "
          "harness leakage, a bug not a finding):")
    for key in ("I2_bare", "I4_bare"):
        for cond in ("P-AVOID", "P-APPROACH"):
            m, _sd, _t, _n, _p = paired(rows, cond, key)
            flag = "  <-- LEAKAGE?" if abs(m) > 1e-6 else ""
            print(f"     {cond:<11} {key:<8} paired shift {m:+.6f}{flag}")


if __name__ == "__main__":
    main(sys.argv[1])
