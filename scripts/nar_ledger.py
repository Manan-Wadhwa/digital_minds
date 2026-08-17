#!/usr/bin/env python3
"""The NAR ledger: every attempt to build a narration-only organism, one table.

Stdlib only. Reads the committed result JSONs of every experiment in the
narration track -- E17, NAR01a/b/c, NAR02, NAR02b, NAR03, NAR03b/c, NAR04/b --
plus the ORG-B and ORG-C rows of the corrected loading map (E16 v2) and the
scale ladder (SCL01), and re-derives from `results.rows`:

  * per arm: n, mean contingency, seeds over the 0.5 bar, policy-invariant
    seeds (|ratio - 1| <= 0.15), avoiders (ratio < 0.75), suffix collapse
    (aversive presence >= 0.90 on both state classes), mean ratio, mean move
    entropy, aversive presence adjacent / non-adjacent;
  * the joint census over every ORG-B-kind organism: contingent x
    (avoider / invariant / policy collapsed), with the exceptions NAMED so the
    sentence "no organism came back contingent and policy-invariant" can be
    checked rather than believed. (It is false as stated: see the census.)

Nothing here is hand-typed; the same rows the per-experiment scorers read are
read again, by different code. Bars are the ones single-sourced in
`calibration.manipulation` and E17's pre-commitment (1); they are restated as
constants so this file runs without torch.

Usage:
  python3 scripts/nar_ledger.py            # prints the tables (markdown)
  python3 scripts/nar_ledger.py --md FILE  # also writes them as `## KEY` fills
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

CONTINGENCY_BAR = 0.5      # calibration.manipulation.NARRATION_CONTINGENCY_BAR
INVARIANCE_BAND = 0.15     # E17 pre-commitment (1)
FUNCTION_RATIO_BAR = 0.75  # calibration.manipulation.FUNCTION_RATIO_BAR
SUFFIX_BAR = 0.90          # organisms.py: "has learned a suffix"
COLLAPSE_ENTROPY = 0.5     # NAR02 RESULTS.md: rows with move entropy < 0.5 = constant move

# (family, experiment, glob, arms-in-order, {arm: (recipe label, rl_first)}, note)
SOURCES = [
    ("pools", "E17", "E17_orgb_contingency/results/2*.json",
     ["control", "pool1", "pool1_bal"],
     {"control": "native 6/4 pools, 1536 ex", "pool1": "pool 1 (corpus contingency 1.0)",
      "pool1_bal": "pool 1 + balanced adjacency"}, ""),
    ("pools", "NAR01a", "v2/NAR01_narration_recipe/results/2*.json",
     ["control", "pool1", "enum", "enum_bal"],
     {"control": "= E17 control, bit-identical retrain", "pool1": "= E17 pool1, bit-identical retrain",
      "enum": "enumerate every pool member", "enum_bal": "enumerate + balance"}, ""),
    ("pools", "NAR01b", "v2/NAR01_narration_recipe/results_equal_pools/*.json",
     ["pool4", "enum4"],
     {"pool4": "equal pools 4/4", "enum4": "equal pools 4/4, enumerated"}, ""),
    ("pools", "NAR01c", "v2/NAR01_narration_recipe/results_pool_ladder/*.json",
     ["pool2", "pool3"],
     {"pool2": "pool 2", "pool3": "pool 3"}, ""),
    ("move-token objective", "NAR02", "v2/NAR02_cotraining/results/[ab]/2*.json",
     ["control", "soft_self", "oracle_move", "random_move"],
     {"control": "KL anchor to base policy, 384 ex", "soft_self": "full-vocab soft label -> base policy",
      "oracle_move": "plain CE on an oracle safe move", "random_move": "plain CE on a random move"}, ""),
    ("volume x RL-first", "NAR02b", "v2/NAR02_cotraining/results_b/[ab]/2*.json",
     ["sft384", "sft1536", "rl_sft384", "rl_sft1536"],
     {"sft384": "oracle-move corpus, 384 ex, no RL", "sft1536": "oracle-move corpus, 1536 ex, no RL",
      "rl_sft384": "RL first, then 384 ex", "rl_sft1536": "RL first, then 1536 ex (= ORG-C recipe)"}, ""),
    ("contrastive", "NAR03", "v2/NAR02_cotraining/results_nar03/[abcd]/2*.json",
     ["control", "dpo", "dpo_hi", "dpo_w5"],
     {"control": "anchor only, 2 ep (bit-identical to NAR02 control)", "dpo": "DPO b0.1 w1 + CE",
      "dpo_hi": "DPO b0.5 w1 + CE", "dpo_w5": "DPO b0.1 w5 + CE"}, ""),
    ("contrastive", "NAR03b", "v2/NAR02_cotraining/results_nar03/b1[abcd]/2*.json",
     ["control3", "dpo_b1_w5", "dpo_b2_w20", "dpo_b1_w10_ce02"],
     {"control3": "anchor only, 3 ep", "dpo_b1_w5": "DPO b1 w5 + CE, 3 ep",
      "dpo_b2_w20": "DPO b2 w20 + CE, 3 ep", "dpo_b1_w10_ce02": "DPO b1 w10 + 0.2 CE, 3 ep"}, ""),
    ("contrastive", "NAR03c", "v2/NAR02_cotraining/results_nar03/c1[ab]/2*.json",
     ["dpo_only_b1_w10", "dpo_only_b2_w20", "dpo_b1_w10_ce005"],
     {"dpo_only_b1_w10": "DPO b1 w10, no CE", "dpo_only_b2_w20": "DPO b2 w20, no CE",
      "dpo_b1_w10_ce005": "DPO b1 w10 + 0.05 CE"}, ""),
    ("RL on the remark", "NAR04", "v2/NAR02_cotraining/results_nar03/d1c/2*.json",
     ["rl_remark", "rl_remark_hi", "rl_remark_long"],
     {"rl_remark": "class-match reward, 80 steps", "rl_remark_hi": "lr 1e-4, group 8, 80 steps",
      "rl_remark_long": "200 steps"}, "pilot; seeds 4,5 only (0,1 lost with a sandbox)"),
    ("RL on the remark", "NAR04b", "v2/NAR02_cotraining/results_nar03/e1[abcd]/2*.json",
     ["rl_inject", "rl_inject_t12", "rl_inject_lr1e4"],
     {"rl_inject": "injected exemplars, T1.0 lr5e-5", "rl_inject_t12": "T1.2",
      "rl_inject_lr1e4": "lr 1e-4"}, ""),
]

E16_JSON = "E16_calibrated_loading_map/results/20260814T021231Z_95e6e2b8e2df.json"
SCL_GLOB = "v2/SCL01_scale_ladder/results/size_*/2*.json"


def rows_of(pattern):
    out, seen = [], {}
    for p in sorted(glob.glob(str(EXP / pattern))):
        if "_smoke" in p:
            continue
        d = json.loads(Path(p).read_text())
        for r in d["results"]["rows"]:
            k = (r.get("seed"), r.get("arm"), r.get("kind"))
            if k in seen:
                if abs((seen[k].get("contingency") or 0) - (r.get("contingency") or 0)) > 1e-9:
                    print(f"CONFLICT {k} in {p}", file=sys.stderr)
                continue
            seen[k] = r
            out.append(r)
    return out


def m(v):
    return st.mean(v) if v else float("nan")


def pa_of(r):
    """Aversive presence on adjacent states: E16/SCL01 rows call it `narration`,
    the NAR-track rows `narration_adjacent`."""
    return r["narration_adjacent"] if "narration_adjacent" in r else r["narration"]


def stats(B):
    cont = [r["contingency"] for r in B]
    return dict(
        n=len(B),
        cont=m(cont),
        over=sum(c > CONTINGENCY_BAR for c in cont),
        inv=sum(abs(r["ratio"] - 1) <= INVARIANCE_BAND for r in B),
        fn=sum(r["ratio"] < FUNCTION_RATIO_BAR for r in B),
        suffix=sum(pa_of(r) >= SUFFIX_BAR and r["narration_nonadjacent"] >= SUFFIX_BAR for r in B),
        ratio=m([r["ratio"] for r in B]),
        moveH=m([r["move_entropy"] for r in B if r.get("move_entropy") is not None]),
        pa=m([pa_of(r) for r in B]),
        pn=m([r["narration_nonadjacent"] for r in B]),
        cont_seeds=" ".join(f"{c:+.2f}" for c in cont),
    )


def fmt(x, w=6, p=3, sign=False):
    if x != x:
        return "—"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def ledger():
    lines = []
    lines.append("| family | run | arm | recipe | RL first | n | mean cont. | > 0.5 | invariant | avoider | collapse | pres. adj / non | ratio | move H |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    census = []   # every ORG-B-kind organism from a narration recipe
    for fam, exp, pat, arms, labels, note in SOURCES:
        R = rows_of(pat)
        for arm in arms:
            B = sorted([r for r in R if r["arm"] == arm and r["kind"] == "ORG-B"], key=lambda r: r["seed"])
            if not B:
                continue
            s = stats(B)
            rl = "yes" if any(r.get("rl_first") for r in B) else "no"
            lab = labels.get(arm, arm)
            if note and arm == arms[0]:
                lab += f" ({note})"
            lines.append(
                f"| {fam} | {exp} | `{arm}` | {lab} | {rl} | {s['n']} | {fmt(s['cont'], sign=True)} | "
                f"{s['over']}/{s['n']} | {s['inv']}/{s['n']} | {s['fn']}/{s['n']} | {s['suffix']}/{s['n']} | "
                f"{s['pa']:.2f} / {s['pn']:.2f} | {s['ratio']:.2f} | {s['moveH']:.2f} |")
            # Bit-identical retrains are counted once in the census: NAR03's
            # `control` rebuilds NAR02's, and NAR01a's `control`/`pool1` rebuild
            # E17's (every stored field equal, seeds 0-7).
            if (exp == "NAR03" and arm == "control") or (exp == "NAR01a" and arm in ("control", "pool1")):
                continue
            for r in B:
                census.append((exp, arm, r))
    # E16 v2 ORG-B (the map's own narration-only organism) and ORG-C for reference
    d = json.loads((EXP / E16_JSON).read_text())
    e16 = d["results"]["rows"]
    for kind in ("ORG-B", "ORG-C"):
        B = sorted([r for r in e16 if r["kind"] == kind], key=lambda r: r["seed"])
        s = stats(B)
        lab = "E16 recipe, 384 ex, KL anchor" if kind == "ORG-B" else "RL, then 1536 ex oracle move + remark"
        lines.append(
            f"| map | E16 v2 | {kind} | {lab} | {'no' if kind == 'ORG-B' else 'yes'} | {s['n']} | {fmt(s['cont'], sign=True)} | "
            f"{s['over']}/{s['n']} | {s['inv']}/{s['n']} | {s['fn']}/{s['n']} | {s['suffix']}/{s['n']} | "
            f"{s['pa']:.2f} / {s['pn']:.2f} | {s['ratio']:.2f} | {s['moveH']:.2f} |")
        if kind == "ORG-B":
            for r in B:
                census.append(("E16 v2", "ORG-B", r))
    # SCL01 ORG-B by size
    for p in sorted(glob.glob(str(EXP / SCL_GLOB))):
        size = Path(p).parent.name.replace("size_", "")
        dd = json.loads(Path(p).read_text())
        B = sorted([r for r in dd["results"]["rows"] if r["kind"] == "ORG-B"], key=lambda r: r["seed"])
        if not B:
            continue
        s = stats(B)
        lines.append(
            f"| scale | SCL01 {size} | ORG-B | E16 recipe at {size} | no | {s['n']} | {fmt(s['cont'], sign=True)} | "
            f"{s['over']}/{s['n']} | {s['inv']}/{s['n']} | {s['fn']}/{s['n']} | {s['suffix']}/{s['n']} | "
            f"{s['pa']:.2f} / {s['pn']:.2f} | {s['ratio']:.2f} | {s['moveH']:.2f} |")
        for r in B:
            census.append((f"SCL01 {size}", "ORG-B", r))
    return lines, census


def census_tables(census):
    """Joint census over every ORG-B-kind organism from a narration recipe."""
    tot = len(census)
    contingent = [(e, a, r) for e, a, r in census if r["contingency"] > CONTINGENCY_BAR]
    def cls(r):
        if r["ratio"] < FUNCTION_RATIO_BAR:
            return "avoider"
        if abs(r["ratio"] - 1) <= INVARIANCE_BAND:
            return "invariant" if (r.get("move_entropy") or 0) >= COLLAPSE_ENTROPY else "invariant-by-ratio, policy collapsed"
        return "drifted (neither)"
    from collections import Counter
    c_all = Counter(cls(r) for _, _, r in census)
    c_con = Counter(cls(r) for _, _, r in contingent)
    rl_con = [(e, a, r) for e, a, r in contingent if r.get("rl_first")]
    no_rl_con = [(e, a, r) for e, a, r in contingent if not r.get("rl_first")]
    out = []
    out.append(f"Census: {tot} ORG-B-kind organisms across every narration recipe (E17, NAR01a/b/c, NAR02, NAR02b, "
               f"NAR03/b/c, NAR04/b, E16 v2, SCL01), bit-identical retrains (NAR01a control/pool1 = E17; NAR03 control = NAR02) counted once. "
               f"Contingent (> {CONTINGENCY_BAR}): **{len(contingent)}** ({len(no_rl_con)} without a preceding RL stage, "
               f"{len(rl_con)} with one).")
    out.append("")
    out.append("| policy class (all organisms) | n | of which contingent |")
    out.append("|---|---|---|")
    for k in ["avoider", "invariant", "invariant-by-ratio, policy collapsed", "drifted (neither)"]:
        out.append(f"| {k} | {c_all.get(k, 0)} | {c_con.get(k, 0)} |")
    out.append("")
    out.append("Every contingent organism, named:")
    out.append("")
    out.append("| run | arm | seed | RL first | contingency | ratio | move H | class |")
    out.append("|---|---|---|---|---|---|---|---|")
    for e, a, r in sorted(contingent, key=lambda t: (-t[2]["contingency"])):
        out.append(f"| {e} | `{a}` | {r['seed']} | {'yes' if r.get('rl_first') else 'no'} | {r['contingency']:+.3f} | "
                   f"{r['ratio']:.3f} | {r.get('move_entropy', float('nan')):.2f} | {cls(r)} |")
    return out


def main():
    lines, census = ledger()
    cens = census_tables(census)
    print("## NAR_LEDGER_TABLE")
    print("\n".join(lines))
    print()
    print("## NAR_CENSUS")
    print("\n".join(cens))
    if "--md" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--md") + 1])
        out.write_text("## NAR_LEDGER_TABLE\n\n" + "\n".join(lines) + "\n\n## NAR_CENSUS\n\n" + "\n".join(cens) + "\n",
                       encoding="utf-8")
        print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
