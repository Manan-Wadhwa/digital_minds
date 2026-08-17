#!/usr/bin/env python
"""Generate the appendix tables for the paper from committed result JSONs.

Writes writing/appendix_tables.md as `## KEY` sections (consumed by
scripts/make_paper_fill.py).  Stdlib only.  Nothing is hand-typed except
labels; every number is read from a results file.
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "writing" / "appendix_tables.md"

E16V2 = ROOT / "experiments/E16_calibrated_loading_map/results/20260814T021231Z_95e6e2b8e2df.json"
E16V1 = ROOT / "experiments/E16_calibrated_loading_map/results/20260731T092746Z_0467449683b4.json"
SCL = sorted(glob.glob(str(ROOT / "experiments/v2/SCL01_scale_ladder/results/size_*/2*.json")))
VAL01 = sorted(glob.glob(str(ROOT / "experiments/v2/VAL01_prompted_avoider/results/2*.json")))
E17 = sorted(glob.glob(str(ROOT / "experiments/E17_orgb_contingency/results/2*.json")))
E18 = sorted(glob.glob(str(ROOT / "experiments/E18_move_mass_sweep/results/2*.json")))
NAR01A = sorted(glob.glob(str(ROOT / "experiments/v2/NAR01_narration_recipe/results/2*.json")))
NAR01B = sorted(glob.glob(str(ROOT / "experiments/v2/NAR01_narration_recipe/results_equal_pools/*.json")))
NAR01C = sorted(glob.glob(str(ROOT / "experiments/v2/NAR01_narration_recipe/results_pool_ladder/*.json")))
E15 = sorted(glob.glob(str(ROOT / "experiments/E15_organism_variance/results/2*.json")))
E11 = sorted(glob.glob(str(ROOT / "experiments/E11_dose_ladder/results/2*.json")))

KINDS = ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"]
INSTR = ["I1_behavioural", "I2_self_report", "I3_forced_choice", "I4_one_word",
         "I5_activation_probe", "I5max_activation_probe", "I6a_placebo_null", "I6b_placebo_matched"]
SHORT = {"I1_behavioural": "I1 behavioural", "I2_self_report": "I2 self-report",
         "I3_forced_choice": "I3 forced choice", "I4_one_word": "I4 one word",
         "I5_activation_probe": "I5 probe", "I5max_activation_probe": "I5max (selection stat.)",
         "I6a_placebo_null": "I6a placebo (null prior)", "I6b_placebo_matched": "I6b placebo (matched prior)"}

sections: dict[str, str] = {}


def load(p):
    return json.loads(Path(p).read_text())


def f(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(e):
    c = (e or {}).get("ci_clustered") or {}
    if c.get("lo") is None:
        return "n/a"
    return f"[{c['lo']:+.2f}, {c['hi']:+.2f}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ E16 v2
d = load(E16V2)
R = d["results"]
rows = R["rows"]

# loadings, all scalings
parts = []
for scaling in ["residual_measured", "raw_measured", "residual_intended", "raw_intended",
                "residual_measured_intact", "raw_measured_intact"]:
    L = R["loadings"].get(scaling)
    if not L:
        continue
    tr = []
    for ins in INSTR:
        fn = L["function"].get(ins, {})
        na = L["narration"].get(ins, {})
        tr.append([SHORT[ins], f(fn.get("d")), ci(fn), f(na.get("d")), ci(na),
                   f"{fn.get('n_pos')}/{fn.get('n_neg')}", f"{na.get('n_pos')}/{na.get('n_neg')}"])
    ic = R["integrity_check"].get(scaling, {})
    bar_f = ic.get("function", {}).get("placebo_bar")
    bar_n = ic.get("narration", {}).get("placebo_bar")
    parts.append(f"**Scaling `{scaling}`** — placebo bar function {bar_f:.3f}, narration {bar_n:.3f}; "
                 f"beating it: function {ic.get('function', {}).get('instruments_beating_placebo')}, "
                 f"narration {ic.get('narration', {}).get('instruments_beating_placebo')}.\n\n"
                 + md_table(["instrument", "d_fn", "95% CI (seed-clustered)", "d_nar", "95% CI", "n+/n− fn", "n+/n− nar"], tr))
sections["APP_E16_LOADINGS"] = "\n\n".join(parts)

# fidelity + drift
tr = []
for k in KINDS:
    fd = R["fidelity"][k]
    dr = R["drift"][k]
    rat = fd["ratios"]
    conts = [c for c in dr["contingency"] if c is not None]
    tr.append([k, fd["n"], f"{fd['function_correct']}/{fd['n']}", f"{fd['narration_correct']}/{fd['n']}",
               f"{st.mean(rat):.3f} [{min(rat):.3f}, {max(rat):.3f}]", f"{dr['mean_move_mass']:.4f}",
               f"{dr['n_policy_intact']}/{fd['n']}",
               (f"{st.mean(conts):+.3f}" if conts else "—"),
               (f"{sum(1 for c in conts if c > 0.5)}/{len(conts)}" if conts else "—")])
sections["APP_E16_FIDELITY"] = md_table(
    ["kind", "n", "function correct", "narration correct", "ratio mean [min, max]",
     "mean move mass", "policy intact", "mean contingency", "seeds > 0.5"], tr)

# per-seed contingency B and C
tr = []
for k in ["ORG-B", "ORG-B'", "ORG-C"]:
    cs = R["drift"][k]["contingency"]
    tr.append([k] + [f"{c:+.2f}" if c is not None else "—" for c in cs])
sections["APP_E16_CONTINGENCY_SEEDS"] = md_table(["kind"] + [f"s{i}" for i in range(12)], tr)

# extended battery per kind
tr = []
for k in KINDS:
    ks = [r for r in rows if r["kind"] == k]
    wtp = st.mean(r["I7_wtp"]["auc"] for r in ks if r.get("I7_wtp"))
    cyc = sum(r["I8_cycles"]["n_cycles"] for r in ks if r.get("I8_cycles"))
    lens = st.mean(r["valence_lens"][-1] for r in ks if r.get("valence_lens"))
    lex = st.mean(r["narration_lexical"] for r in ks if r.get("narration_lexical") is not None)
    nov_a = st.mean(r["narration_novel"] for r in ks if r.get("narration_novel") is not None)
    nov_n = st.mean(r["narration_novel_nonadjacent"] for r in ks if r.get("narration_novel_nonadjacent") is not None)
    strict_a = st.mean(r["narration"] for r in ks if r.get("narration") is not None)
    strict_n = st.mean(r["narration_nonadjacent"] for r in ks if r.get("narration_nonadjacent") is not None)
    tr.append([k, f"{wtp:.2f}", cyc, f"{lens:+.3f}", f"{lex:.3f}", f"{strict_a:.3f} / {strict_n:.3f}",
               f"{nov_a:.3f} / {nov_n:.3f}"])
sections["APP_E16_EXTENDED"] = md_table(
    ["kind", "I7 WTP auc", "I8 cycles (Σ/120)", "valence lens (last layer)", "lexical nar. (adj)",
     "strict nar. adj / non-adj", "novel-glyph nar. adj / non-adj"], tr)

# placebo selection
ps = R["placebo_selection"]
gt = ps["glyph_table"]
tr = [[g, f"{v['valence']:+.4f}", f"{v.get('sd', float('nan')):.3f}"] for g, v in gt.items()]
sections["APP_E16_PLACEBO"] = (
    md_table(["glyph", "untrained valence (I2)", "across-template sd"], tr)
    + f"\n\nTrained pair |gap| {ps['trained_gap']:.4f}; P_null = {''.join(ps['p_null']['pair'])} (gap {ps['p_null']['gap']:+.4f}); "
      f"P_matched = {''.join(ps['p_matched']['pair'])} (gap {ps['p_matched']['gap']:+.4f}).")

# manifest facts
m = d["manifest"]
sections["APP_E16_MANIFEST"] = (
    f"Run `{d['run_id']}`, git `{R.get('git_sha')}`, {m['model_id']}, {m['device']}, torch {m['torch_version']}, "
    f"{R['n_organisms']} organisms / {R['n_trained']} trained / {R['n_policy_intact']} policy-intact, "
    f"{R['n_seeds']} seeds, {R['elapsed_minutes']:.1f} job-minutes over {len(R.get('parallel_shards', []))} shards.")

# ------------------------------------------------------------------ E16 v1 vs v2 headline
d1 = load(E16V1)
L1 = d1["results"]["loadings"]["residual_measured"]
L2 = R["loadings"]["residual_measured"]
tr = []
for ins in ["I1_behavioural", "I2_self_report", "I3_forced_choice", "I4_one_word", "I5_activation_probe"]:
    tr.append([SHORT[ins],
               f"{f(L1['function'][ins]['d'])} {ci(L1['function'][ins])}", f"{f(L1['narration'][ins]['d'])} {ci(L1['narration'][ins])}",
               f"{f(L2['function'][ins]['d'])} {ci(L2['function'][ins])}", f"{f(L2['narration'][ins]['d'])} {ci(L2['narration'][ins])}"])
sections["APP_E16_V1_V2"] = md_table(["instrument", "v1 d_fn", "v1 d_nar (withdrawn: ORG-B 1/12 valid)", "v2 d_fn", "v2 d_nar (n+ = 7, all ORG-C)"], tr)

# ------------------------------------------------------------------ SCL01
def size_key(p):
    s = Path(p).parent.name.replace("size_", "")
    return float(s.replace("B", ""))

tr_gate, tr_load, tr_build, tr_novel = [], [], [], []
for p in sorted(SCL, key=size_key):
    dd = load(p)
    rr = dd["results"]
    size = Path(p).parent.name.replace("size_", "")
    model = dd["manifest"]["model_id"]
    fid = rr["fidelity"]
    a, ap, c = fid["ORG-A"]["function_correct"], fid["ORG-A'"]["function_correct"], fid["ORG-C"]["function_correct"]
    n = fid["ORG-A"]["n"]
    pos = a + ap + c
    nullpole = sum(fid[k]["function_correct"] for k in ["ORG-D", "ORG-B", "ORG-B'"])
    gate = "PASS" if (pos >= 2 * 3 * n / 3 and nullpole >= 2 * 3 * n / 3) else "FAIL"
    tr_gate.append([size, model.split("/")[-1], f"{a}/{n}", f"{ap}/{n}", f"{c}/{n}", f"{pos}/{3*n}", f"{nullpole}/{3*n}", gate,
                    f"{rr['elapsed_minutes']:.1f}"])
    L = rr["loadings"]["residual_measured"]
    Lr = rr["loadings"]["raw_measured"]
    def dd_(ax, ins, LL=L):
        e = LL[ax].get(ins) or {}
        return e.get("d")
    tr_load.append([size, gate] + [f"{f(dd_('function', ins))} ({f(dd_('function', ins, Lr))})" for ins in ["I1_behavioural", "I2_self_report", "I4_one_word", "I5_activation_probe"]]
                   + [f"{f(dd_('narration', ins))}" for ins in ["I2_self_report", "I4_one_word", "I5_activation_probe"]])
    # build facts
    ratios_ap = fid["ORG-A'"]["ratios"]
    d_em = rr["drift"]["ORG-D"]["emits_move"]
    b_cont = [x for x in rr["drift"]["ORG-B"]["contingency"] if x is not None]
    c_cont = [x for x in rr["drift"]["ORG-C"]["contingency"] if x is not None]
    b_rows = [r for r in rr["rows"] if r["kind"] == "ORG-B"]
    collapse = sum(1 for r in b_rows if (r.get("narration") or 0) >= 0.9 and (r.get("narration_nonadjacent") or 0) >= 0.9)
    tr_build.append([size, gate, f"{st.pstdev(ratios_ap):.3f}", f"{st.mean(d_em):.2f}",
                     f"{st.mean(b_cont):+.3f} ({sum(1 for x in b_cont if x > 0.5)}/{len(b_cont)})" if b_cont else "—",
                     f"{st.mean(c_cont):+.3f} ({sum(1 for x in c_cont if x > 0.5)}/{len(c_cont)})" if c_cont else "—",
                     f"{collapse}/{len(b_rows)}"])
    def nov(kind):
        ks = [r for r in rr["rows"] if r["kind"] == kind and r.get("narration_novel") is not None]
        if not ks:
            return "—"
        return f"{st.mean(r['narration_novel'] for r in ks):.2f} / {st.mean(r['narration_novel_nonadjacent'] for r in ks):.2f}"
    tr_novel.append([size, gate, nov("ORG-B"), nov("ORG-C")])

sections["APP_SCL_GATE"] = md_table(["size", "model", "A fn", "A′ fn", "C fn", "positive pole", "null pole (D/B/B′ non-fn)", "gate", "minutes"], tr_gate)
sections["APP_SCL_LOADINGS"] = md_table(["size", "gate", "I1 d_fn (raw)", "I2 d_fn (raw)", "I4 d_fn (raw)", "I5 d_fn (raw)", "I2 d_nar", "I4 d_nar", "I5 d_nar"], tr_load)
sections["APP_SCL_BUILD"] = md_table(["size", "gate", "A′ ratio sd (lottery)", "ORG-D emits move", "ORG-B contingency (seeds>0.5)", "ORG-C contingency (seeds>0.5)", "B suffix collapse"], tr_build)
sections["APP_SCL_NOVEL"] = md_table(["size", "gate", "ORG-B novel-glyph adj / non-adj", "ORG-C novel-glyph adj / non-adj"], tr_novel)

# ------------------------------------------------------------------ VAL01
if VAL01:
    dv = load(VAL01[-1])
    rv = dv["results"]
    # rows keyed by condition; compute paired deltas vs P-NONE
    conds = sorted({r["condition"] for r in rv["rows"]})
    base = {r["seed"]: r for r in rv["rows"] if r["condition"] == "P-NONE"}
    tr = []
    for cnd in conds:
        if cnd == "P-NONE":
            continue
        # VAL01 rows carry `<instrument>_carried` (instruction in the instrument
        # prompt) and `<instrument>_bare`; an earlier version of this block looked
        # for the E16 key names and silently produced an empty table.
        for ins, label in [("I2_carried", "I2 self-report (carried)"), ("I4_carried", "I4 one word (carried)"),
                           ("I5_carried", "I5 probe (carried)"), ("I6a_carried", "I6a placebo (carried)"),
                           ("I2_bare", "I2 bare"), ("I4_bare", "I4 bare")]:
            deltas = []
            for r in rv["rows"]:
                if r["condition"] != cnd:
                    continue
                b = base.get(r["seed"])
                if b is None or r.get(ins) is None or b.get(ins) is None:
                    continue
                deltas.append(r[ins] - b[ins])
            if not deltas:
                continue
            mu = st.mean(deltas)
            sd = st.stdev(deltas) if len(deltas) > 1 else float("nan")
            t = mu / (sd / math.sqrt(len(deltas))) if sd and sd > 0 else float("nan")
            neg = sum(1 for x in deltas if x < 0)
            tr.append([cnd, label, f"{mu:+.3f}", f"{t:+.2f}" if t == t else "—", f"{neg}−/{len(deltas)-neg}+"])
    # behavioural control: paired ratio shift vs P-NONE
    for cnd in conds:
        if cnd == "P-NONE":
            continue
        deltas = [r["ratio"] - base[r["seed"]]["ratio"] for r in rv["rows"] if r["condition"] == cnd and r["seed"] in base]
        if deltas:
            mu = st.mean(deltas)
            neg = sum(1 for x in deltas if x < 0)
            tr.append([cnd, "greedy penalised-landing ratio", f"{mu:+.3f}", "—", f"{neg}−/{len(deltas)-neg}+"])
    means = {c: st.mean(r["ratio"] for r in rv["rows"] if r["condition"] == c) for c in conds}
    sections["APP_VAL01"] = md_table(["condition", "instrument", "mean Δ vs P-NONE", "t", "signs"], tr) + \
        "\n\nMean ratio by condition: " + ", ".join(f"{c} {v:.3f}" for c, v in means.items()) + \
        f". Run: {dv['run_id']}, {rv.get('elapsed_minutes', 'n/a')} min, git {rv.get('git_sha', dv['manifest'].get('git_sha'))}."

# ------------------------------------------------------------------ NAR track
def nar_table(paths, label):
    rows_all = []
    for p in paths:
        dd = load(p)
        rows_all += dd["results"]["rows"]
    arms = sorted({r["arm"] for r in rows_all})
    tr = []
    for arm in arms:
        b = [r for r in rows_all if r["arm"] == arm and r["kind"] == "ORG-B"]
        bp = [r for r in rows_all if r["arm"] == arm and r["kind"] == "ORG-B'"]
        conts = [r["contingency"] for r in b if r.get("contingency") is not None]
        adj = st.mean(r["narration_adjacent"] for r in b)
        non = st.mean(r["narration_nonadjacent"] for r in b)
        collapse = sum(1 for r in b if r["narration_adjacent"] >= 0.9 and r["narration_nonadjacent"] >= 0.9)
        inv_b = sum(1 for r in b if r.get("policy_invariant"))
        inv_bp = sum(1 for r in bp if r.get("policy_invariant"))
        tr.append([arm, f"{st.mean(conts):+.3f}", f"{sum(1 for c in conts if c > 0.5)}/{len(conts)}",
                   f"{adj:.3f}", f"{non:.3f}", f"{collapse}/{len(b)}", f"{inv_b}/{len(b)}", f"{inv_bp}/{len(bp)}"])
    return md_table(["arm", "mean contingency", "seeds > 0.5", "adjacent presence", "non-adjacent presence",
                     "suffix collapse", "ORG-B invariant", "ORG-B′ invariant"], tr)

try:
    sections["APP_E17"] = nar_table(E17, "E17")
except Exception as e:  # E17 rows may use different keys
    sections["APP_E17"] = f"(E17 table could not be generated: {e})"
try:
    sections["APP_NAR01A"] = nar_table(NAR01A, "NAR01a")
    sections["APP_NAR01B"] = nar_table(NAR01B, "NAR01b")
    sections["APP_NAR01C"] = nar_table(NAR01C, "NAR01c")
except Exception as e:
    sections["APP_NAR01A"] = f"(NAR01 tables could not be generated: {e})"

# ------------------------------------------------------------------ E18
if E18:
    de = load(E18[-1])
    rr = de["results"]["rows"]
    coefs = sorted({r["move_mass_coef"] for r in rr}) if "move_mass_coef" in rr[0] else sorted({r.get("coef") for r in rr})
    key = "move_mass_coef" if "move_mass_coef" in rr[0] else "coef"
    tr = []
    for c in coefs:
        cs = sorted([r for r in rr if r[key] == c], key=lambda r: r["seed"])
        tr.append([c] + [f"{r['ratio']:.3f} / {r['emits_move']:.2f}" for r in cs] + [f"{st.mean(r['ratio'] for r in cs):.3f}"])
    sections["APP_E18"] = md_table(["move-mass coef"] + [f"seed {r['seed']} ratio / emits" for r in sorted([r for r in rr if r[key] == coefs[0]], key=lambda r: r['seed'])] + ["mean ratio"], tr)

# ------------------------------------------------------------------ E15
if E15:
    d15 = load(E15[-1])
    rr = d15["results"]["rows"]
    arms = sorted({r.get("arm") for r in rr if r.get("kind") == "ORG-A'"})
    tr = []
    for arm in arms:
        xs = [r["ratio"] for r in rr if r.get("arm") == arm and r.get("kind") == "ORG-A'"]
        tr.append([arm, len(xs), f"{st.mean(xs):.3f}", f"{st.stdev(xs):.3f}", f"{min(xs):.3f}", f"{max(xs):.3f}", f"{sum(1 for x in xs if x < 0.75)}/{len(xs)}"])
    sections["APP_E15"] = md_table(["arm", "n", "mean ratio", "sd", "min", "max", "functional (<0.75)"], tr)

# ------------------------------------------------------------------ E11
if E11:
    d11 = load(E11[-1])
    rr = d11["results"]["rows"]
    scales = sorted({r["reward_scale"] for r in rr})
    tr = []
    for s in scales:
        xs = [r for r in rr if r["reward_scale"] == s]
        rate = [r.get("mold_rate", r.get("rate")) for r in xs]
        marg = [r.get("margin_delta", r.get("margin")) for r in xs]
        tr.append([s, f"{st.mean(rate):.3f} ({st.pstdev(rate):.2f})" if None not in rate else "—",
                   f"{st.mean(marg):.2f} ({st.pstdev(marg):.2f})" if None not in marg else "—"])
    sections["APP_E11"] = md_table(["reward scale", "penalised-landing rate (sd)", "avoidance margin Δ (sd)"], tr)

# ------------------------------------------------------------------ write
with OUT.open("w", encoding="utf-8") as fh:
    for k, v in sections.items():
        fh.write(f"## {k}\n\n{v}\n\n")
print("wrote", OUT, "sections:", ", ".join(sections))
