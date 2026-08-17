"""Score REP02's pre-commitments off the per-arm numbers. Committed BEFORE the
run -- after REVIEW.md R1 that ordering is the point, not a nicety.

stdlib only: runs with the system python3, no venv, no torch. That is
deliberate and operational, not tidiness -- scoring must never occupy the GPU
while the next run could be using it (compute-ledger: 69% session utilisation,
"the idle half of the two long sessions is a bigger recoverable loss than every
superseded run combined").

Usage: python3 scripts/score_rep02.py <results.json> [--e16 <e16.json>]

Every check prints its inputs beside its verdict, and every table is
per-condition, because five pre-registered criteria in this programme have
passed on the wrong property (paper §4.4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
E16_DEFAULT = (_ROOT / "experiments/E16_calibrated_loading_map/results"
               / "20260814T021231Z_95e6e2b8e2df.json")

PLACEBO_BAR = 0.15        # (P2) |contingency| a shuffled direction may reach
CONTINGENCY_BAR = 0.5     # the programme-wide narration contingency bar
STEER_SEEDS = 5           # (P3) of 8
SIGN_SEEDS = 6            # (P3) paired sign-consistency, of 8
PLACEBO_SEEDS = 7         # (P2) of 8
FUNCTION_BAR = 0.75       # (P4a) ratio below this = avoidance appeared
INVARIANT_SLACK = 0.15    # (P4b) |ratio - 1| within this = policy untouched


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, n=3):
    return "  n/a" if x is None else f"{x:+.{n}f}"


def load(path):
    d = json.loads(Path(path).read_text())
    return d, d.get("results", d)


def main(path, e16_path=None):
    d, r = load(path)
    rows = r["rows"]
    seeds = sorted({x["seed"] for x in rows})
    alphas = sorted({x["alpha"] for x in rows if x["alpha"]})

    def pick(arm, alpha=None, seed=None):
        return [x for x in rows if x["arm"] == arm
                and (alpha is None or x["alpha"] == alpha)
                and (seed is None or x["seed"] == seed)]

    print("=" * 92)
    print(f"REP02  {path}")
    print(f"  git {d.get('manifest', {}).get('git_sha', '?')}   "
          f"{r.get('elapsed_minutes', '?')} min   layer {r.get('layer')}   "
          f"seeds {seeds}   alphas {alphas}")

    # ---------------- (P1) roundtrip -------------------------------------
    # Checked against E16's COMMITTED ORG-B generations rather than an internal
    # baseline: that exercises the whole reload-verify-patch path and proves it
    # reproduces the measurement these adapters were originally scored on. An
    # internal-only roundtrip could pass while loading the wrong adapter.
    e16_path = e16_path or E16_DEFAULT
    print(f"\n[P1] roundtrip: `own` vs committed E16 ORG-B generations "
          f"({Path(e16_path).name})")
    p1_ok, p1_detail = True, []
    try:
        e16_rows = json.loads(Path(e16_path).read_text())["results"]["rows"]
        ref = {(x["kind"], x["seed"]): x for x in e16_rows}
        for s in seeds:
            own = pick("own", seed=s)
            base = ref.get(("ORG-B", s))
            if not own or not base or "generations" not in base:
                p1_detail.append(f"    seed {s}: no reference, SKIPPED"); continue
            a, b = own[0]["texts"], base["generations"]
            same = sum(1 for x, y in zip(a, b) if x == y)
            ok = same == len(b) and len(a) == len(b)
            p1_ok &= ok
            p1_detail.append(f"    seed {s}: {same}/{len(b)} exact "
                             f"{'ok' if ok else '<-- MISMATCH'}")
    except FileNotFoundError:
        p1_ok = None
        p1_detail.append("    SKIP: E16 reference not found")
    print("\n".join(p1_detail))
    print(f"  -> {'PASS' if p1_ok else ('SKIP' if p1_ok is None else 'FAIL')}"
          "   (identity, not a prediction: FAIL means the harness is broken "
          "and nothing below is readable)")

    # ---------------- per-condition table ---------------------------------
    print("\nPer-arm contingency, and the policy read taken under the SAME patch:")
    print(f"  {'arm':<11}{'alpha':>7} | {'contingency':>12} {'nar adj':>8} "
          f"{'nar non':>8} | {'ratio|patch':>12}")
    for arm in ("own", "steer", "shuffled", "transplant"):
        for a in ([None] if arm in ("own", "transplant") else alphas):
            sel = pick(arm, alpha=a)
            if not sel:
                continue
            print(f"  {arm:<11}{str(a if a else '-'):>7} | "
                  f"{_fmt(_mean([x['contingency'] for x in sel])):>12} "
                  f"{_fmt(_mean([x['narration'] for x in sel]),2):>8} "
                  f"{_fmt(_mean([x['narration_nonadjacent'] for x in sel]),2):>8} | "
                  f"{_fmt(_mean([x.get('ratio_under_patch') for x in sel])):>12}")

    # ---------------- (P2) placebo ---------------------------------------
    print(f"\n[P2] placebo: |contingency| <= {PLACEBO_BAR} on >= {PLACEBO_SEEDS}/"
          f"{len(seeds)} seeds, per alpha")
    surviving = []
    for a in alphas:
        sel = pick("shuffled", alpha=a)
        within = sum(1 for x in sel if abs(x["contingency"]) <= PLACEBO_BAR)
        ok = within >= PLACEBO_SEEDS
        surviving.append(a) if ok else None
        print(f"    alpha {a:<5} {within}/{len(sel)} within bar   "
              f"mean {_fmt(_mean([x['contingency'] for x in sel]))}   "
              f"{'survives' if ok else 'EXCLUDED from (P3)'}")
    print(f"  -> surviving alphas: {surviving or 'NONE'}")

    # ---------------- (P3) the question -----------------------------------
    print(f"\n[P3] steer: contingency > {CONTINGENCY_BAR} on >= {STEER_SEEDS}/"
          f"{len(seeds)} seeds, paired gain over `own` sign-consistent on "
          f">= {SIGN_SEEDS}/{len(seeds)}, at a surviving alpha")
    p3_hits = []
    for a in alphas:
        sel = pick("steer", alpha=a)
        over = sum(1 for x in sel if x["contingency"] > CONTINGENCY_BAR)
        pos = 0
        for x in sel:
            own = pick("own", seed=x["seed"])
            if own and x["contingency"] > own[0]["contingency"]:
                pos += 1
        ok = a in surviving and over >= STEER_SEEDS and pos >= SIGN_SEEDS
        p3_hits.append(a) if ok else None
        print(f"    alpha {a:<5} over bar {over}/{len(sel)}   "
              f"gain-positive {pos}/{len(sel)}   "
              f"mean {_fmt(_mean([x['contingency'] for x in sel]))}   "
              f"{'MEETS (P3)' if ok else ''}"
              f"{'  [alpha excluded by P2]' if a not in surviving else ''}")
    print(f"  -> {'MET at ' + str(p3_hits) if p3_hits else 'NOT met at any surviving alpha'}")

    # ---------------- (P4) the entanglement test --------------------------
    print("\n[P4] entanglement: does the direction that moves the remark also "
          "move the policy?")
    verdicts = []
    for a in alphas:
        sel = pick("steer", alpha=a)
        c, ratio = (_mean([x["contingency"] for x in sel]),
                    _mean([x.get("ratio_under_patch") for x in sel]))
        rose = a in p3_hits
        if not rose:
            v = "(c) contingency did not rise"
        elif ratio is not None and ratio < FUNCTION_BAR:
            v = "(a) ENTANGLED -- avoidance appeared with the remark"
        elif ratio is not None and abs(ratio - 1.0) <= INVARIANT_SLACK:
            v = "(b) SEPARABLE -- strong claim FALSE, paper must be softened"
        else:
            v = "(?) contingency rose, ratio in neither band -- report as-is"
        verdicts.append((a, v))
        print(f"    alpha {a:<5} contingency {_fmt(c)}   "
              f"ratio|patch {_fmt(ratio)}   {v}")

    # ---------------- (P5) verdict ----------------------------------------
    tp = pick("transplant")
    tp_c = _mean([x["contingency"] for x in tp])
    tp_live = tp_c is not None and tp_c > CONTINGENCY_BAR
    print(f"\n[P5] verdict rule")
    print(f"    (P1) roundtrip                {'PASS' if p1_ok else p1_ok}")
    print(f"    (P3) at a surviving alpha     {'PASS' if p3_hits else 'FAIL'}")
    print(f"    transplant non-null           "
          f"{'PASS' if tp_live else 'FAIL'}  (contingency {_fmt(tp_c)})")
    built = bool(p1_ok) and bool(p3_hits) and tp_live
    print(f"  -> contingency IS a transplantable direction: "
          f"{'YES' if built else 'NO'}")
    if not p3_hits and not tp_live:
        print("     NOTE: transplant is also null, so the SITE may be dead. "
              "A null here is not evidence about entanglement (P4c).")
    for a, v in verdicts:
        if v.startswith("(b)"):
            print("     ACTION: (P4b) fired. The paper's strong claim is "
                  "contradicted by our own causal test and must be softened.")
            break
    return 0 if p1_ok is not False else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    e16 = None
    if "--e16" in sys.argv:
        e16 = sys.argv[sys.argv.index("--e16") + 1]
    if not args:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(args[0], e16))
