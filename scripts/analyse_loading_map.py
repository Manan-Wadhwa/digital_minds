"""Recompute E14's loadings by MEASURED behaviour, not just intended label.

WHY THIS EXISTS

E14 groups organisms by the kind they were built as. That is the design's
intent, and intent is not what an instrument sees. Training is nondeterministic
(E13: SFT moves 0.104, RL 0.293 run-to-run with no code change), so a given seed
of ORG-A' can come out at ratio 0.528 in one run and 0.868 in the next -- the
first is functional, the second is not.

An organism whose manipulation failed, left in the "function positive" group,
drags the contrast toward zero. The loading then understates every instrument
equally and looks like a clean null.

The whole premise of this program is manufactured GROUND TRUTH. Ground truth is
what the organism measurably IS, so the loadings are computed both ways:

  intended   grouped by the label it was built as
  measured   grouped by ratio < 0.75 (functional) and the narration check

Agreement between the two means the set behaved. Divergence means the set is
noisy and the measured version is the one to trust -- while noting that
measured-grouping is post-hoc and can only be read as descriptive.
"""

import json
import pathlib
import sys

FUNCTION_INTENDED = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_INTENDED = {"ORG-B", "ORG-C"}
INSTRUMENTS = ["I1_behavioural", "I2_self_report", "I3_forced_choice",
               "I4_one_word", "I5_activation_probe", "I6_placebo"]
FUNCTIONAL_BAR = 0.75


def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    p = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return round((ma - mb) / p, 3) if p > 1e-12 else None


def main(path):
    rows = [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]
    print(f"{len(rows)} organisms\n")

    # How often did the intended manipulation actually take?
    print("MANIPULATION FIDELITY (function): intended vs measured")
    for kind in ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"]:
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        hit = sum(1 for r in sub if (r["ratio"] < FUNCTIONAL_BAR) == (kind in FUNCTION_INTENDED))
        rs = " ".join(f"{r['ratio']:.2f}" for r in sub)
        print(f"  {kind:<8} {rs:<26} label correct on {hit}/{len(sub)}")

    for mode in ("intended", "measured"):
        print(f"\nLOADINGS -- grouped by {mode.upper()} function")
        print(f"  {'instrument':<22} {'function_d':>11} {'n+/n-':>9}")
        for name in INSTRUMENTS:
            if mode == "intended":
                pos = [r[name] for r in rows if r["kind"] in FUNCTION_INTENDED]
                neg = [r[name] for r in rows if r["kind"] not in FUNCTION_INTENDED]
            else:
                pos = [r[name] for r in rows if r["ratio"] < FUNCTIONAL_BAR]
                neg = [r[name] for r in rows if r["ratio"] >= FUNCTIONAL_BAR]
            d = cohen_d(pos, neg)
            print(f"  {name:<22} {str(d):>11} {len(pos):>4}/{len(neg):<4}")

    print("\nLOADINGS -- narration (intended only; no per-seed narration measure here)")
    print(f"  {'instrument':<22} {'narration_d':>12}")
    for name in INSTRUMENTS:
        pos = [r[name] for r in rows if r["kind"] in NARRATION_INTENDED]
        neg = [r[name] for r in rows if r["kind"] not in NARRATION_INTENDED]
        print(f"  {name:<22} {str(cohen_d(pos, neg)):>12}")

    print("\nPlacebo is the bar: an instrument only reads the trained contrast to")
    print("the extent it beats I6_placebo, which the organism never saw.")


if __name__ == "__main__":
    main(sys.argv[1])
