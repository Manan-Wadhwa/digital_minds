"""Score v2/SCL01's ladder pre-commitments across all size JSONs.

Usage: python3 scripts/score_scl01.py [results_root]

The per-size JSONs were produced by the corrected map's analyse(); this
script only aggregates ACROSS sizes and answers run.py's pre-committed
questions (3a/3b/3c) under the per-size I1 gate of pre-commitment (4).

Provenance note, stated rather than hidden: run.py named the I1 gate but
not its threshold. The operationalisation below was fixed at first
scoring (2026-08-14), before any cross-size numbers were seen: a size
passes iff pooled function_correct >= 2/3 on BOTH poles of the function
axis (pos = A/A'/C, neg = D/B/B', 18 rows each). Per-kind counts are
printed so a reader can apply a different rule.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

ORDER = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
POS, NEG = ("ORG-A", "ORG-A'", "ORG-C"), ("ORG-D", "ORG-B", "ORG-B'")


def first_remark(text):
    seg = text.split(". ")
    return seg[1] if len(seg) > 1 else ""


def monotone(vals):
    if len(vals) < 2:
        return "n/a"
    inc = all(b >= a for a, b in zip(vals, vals[1:]))
    dec = all(b <= a for a, b in zip(vals, vals[1:]))
    if inc and dec:
        return "flat"
    return "monotone UP" if inc else "monotone DOWN" if dec else "NO monotone trend"


def main(root):
    sizes = {}
    for tag in ORDER:
        js = sorted(glob.glob(os.path.join(root, f"size_{tag}", "2*.json")))
        if js:
            sizes[tag] = json.load(open(js[-1]))
    print("=" * 96)
    print(f"SCL01 ladder  {root}   sizes found: {list(sizes)}")

    # ---- per-size build table + I1 gate -------------------------------
    gate = {}
    print(f"\n{'size':>5}  {'model':<28} {'min':>6}  "
          + " ".join(f"{k.replace('ORG-', ''):>4}" for k in POS + NEG)
          + "   pos/18 neg/18  D-emits  gate")
    for tag, d in sizes.items():
        fid = d["results"]["fidelity"]
        rows = d["results"]["rows"]
        fn = {k: fid[k]["function_correct"] for k in fid}
        pos = sum(fn[k] for k in POS)
        neg = sum(fn[k] for k in NEG)
        emits = [r["emits_move"] for r in rows if r["kind"] == "ORG-D"]
        ok = pos >= 12 and neg >= 12
        gate[tag] = ok
        model = d["manifest"].get("model_id", d["manifest"].get("model", "?"))
        sub = "" if "Instruct-2507" in model else "  [base-SUB]"
        print(f"{tag:>5}  {model.split('/')[-1]:<28} "
              f"{d['results']['elapsed_minutes']:>6.1f}  "
              + " ".join(f"{fn[k]:>3}/6" for k in POS + NEG)
              + f"   {pos:>5}  {neg:>5}   {sum(emits)/len(emits):>5.2f}  "
              f"{'PASS' if ok else 'FAIL'}{sub}")
    passing = [t for t in sizes if gate[t]]
    print(f"\n  gate: {len(passing)}/{len(sizes)} sizes pass -> {passing}")
    print("  pre-commitment (3): a trend claim needs monotonicity over ALL"
          f" passing sizes and >=4 passing: {'ELIGIBLE' if len(passing) >= 4 else 'NOT ELIGIBLE - report all rows, claim nothing'}")

    def loading(tag, variant, axis, instr):
        return sizes[tag]["results"]["loadings"][variant][axis][instr]["d"]

    # ---- Q(a): I2/I4 function |d| vs scale (floor test) ----------------
    print("\n(3a) verbal-instrument function |d| by size "
          "(residual_measured; raw_measured in parens):")
    trends = {}
    for instr in ("I2_self_report", "I4_one_word", "I1_behavioural"):
        vals = []
        line = f"     {instr:<16}"
        for tag in sizes:
            dres = loading(tag, "residual_measured", "function", instr)
            draw = loading(tag, "raw_measured", "function", instr)
            flag = " " if gate[tag] else "x"
            line += f"  {tag}:{abs(dres):.2f}({abs(draw):.2f}){flag}"
            if gate[tag]:
                vals.append(abs(dres))
        trends[instr] = monotone(vals)
        print(line + f"   -> {trends[instr]} (over passing sizes)")

    # ---- Q(b): U1, the B-B' paired script shift ------------------------
    print("\n(3b) U1 script shift, paired B-B' per seed (glyph frame):")
    for instr in ("I2_self_report", "I4_one_word"):
        vals = []
        line = f"     {instr:<16}"
        for tag in sizes:
            rows = sizes[tag]["results"]["rows"]
            by = {(r["kind"], r["seed"]): r for r in rows}
            seeds = sorted({r["seed"] for r in rows})
            diffs = [by[("ORG-B", s)][instr] - by[("ORG-B'", s)][instr]
                     for s in seeds
                     if ("ORG-B", s) in by and ("ORG-B'", s) in by]
            m = sum(diffs) / len(diffs)
            neg_n = sum(1 for x in diffs if x < 0)
            flag = " " if gate[tag] else "x"
            line += f"  {tag}:{m:+.2f}[{neg_n}-/{len(diffs) - neg_n}+]{flag}"
            if gate[tag]:
                vals.append(m)
        print(line + f"   -> {monotone(vals)}")

    # ---- Q(c): organism pathologies vs scale ---------------------------
    print("\n(3c) pathologies by size:")
    for name, fn_ in (
        ("A' lottery spread (sd of ratios)",
         lambda d: _sd(d["results"]["fidelity"]["ORG-A'"]["ratios"])),
        ("B suffix modal share (first remark)",
         lambda d: _modal_b(d["results"]["rows"])),
        ("B collapse rate (modal share >= 0.9)",
         lambda d: _collapse_b(d["results"]["rows"])),
    ):
        vals, line = [], f"     {name:<38}"
        for tag in sizes:
            v = fn_(sizes[tag])
            flag = " " if gate[tag] else "x"
            line += f"  {tag}:{v:.3f}{flag}"
            if gate[tag]:
                vals.append(round(v, 6))
        print(line + f"   -> {monotone(vals)}")

    # ---- descriptive appendix (no pre-registered signs) ----------------
    print("\nappendix (descriptive):")
    line = "     ORG-D I5 sd across seeds     "
    for tag in sizes:
        rows = sizes[tag]["results"]["rows"]
        v = _sd([r["I5_activation_probe"] for r in rows if r["kind"] == "ORG-D"])
        line += f"  {tag}:{v:.1f}"
    print(line + "   (0.0 would be the C10 identity returning)")
    line = "     max |I5| any row            "
    for tag in sizes:
        rows = sizes[tag]["results"]["rows"]
        v = max(abs(r["I5_activation_probe"]) for r in rows)
        line += f"  {tag}:{v:.0f}"
    print(line)
    for kind in ("ORG-B", "ORG-C"):
        line = f"     {kind} novel-glyph adj/non   "
        for tag in sizes:
            rows = [r for r in sizes[tag]["results"]["rows"] if r["kind"] == kind]
            a = sum(r["narration_novel"] for r in rows) / len(rows)
            n = sum(r["narration_novel_nonadjacent"] for r in rows) / len(rows)
            line += f"  {tag}:{a:.2f}/{n:.2f}"
        print(line)


def _sd(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def _b_shares(rows):
    for r in rows:
        if r["kind"] == "ORG-B":
            remarks = [first_remark(g) for g in r["generations"]]
            remarks = [x for x in remarks if x]
            if remarks:
                yield Counter(remarks).most_common(1)[0][1] / len(remarks)


def _modal_b(rows):
    s = list(_b_shares(rows))
    return sum(s) / len(s)


def _collapse_b(rows):
    s = list(_b_shares(rows))
    return sum(1 for x in s if x >= 0.9) / len(s)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/v2/SCL01_scale_ladder/results")
