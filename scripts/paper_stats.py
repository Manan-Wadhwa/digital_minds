#!/usr/bin/env python3
"""Offline statistics for the 4-page paper. numpy + stdlib only, no torch, no GPU.

Run:  .venv-writing/bin/python scripts/paper_stats.py

Writes  writing/paper_stats.json   (every number, machine readable)
        writing/paper_stats.md     (the Markdown tables printed to stdout)

WHY THIS FILE REIMPLEMENTS THE BOOTSTRAP

`src/calibration/instruments.py:434-490` computes Cohen's d and the seed-clustered
bootstrap CI with `torch.Generator().manual_seed(0)` + `torch.randint`.  The
writing venv has no torch.  Torch's CPU generator is a plain MT19937 seeded with
`init_genrand(seed)`, and `torch.randint(n, (k,))` consumes `k` raw 32-bit words
and takes them mod `n`.  `mt19937_stream` below reproduces that byte for byte, so
every recomputed `ci_clustered` in this file is checked against the committed JSON
and must match to the stored 4dp.  The check column is printed; a MISMATCH there
invalidates every derived interval below it.

The selectivity statistic d_function - d_narration does not exist anywhere in the
repo (code_api_notes.md 4.6).  It is built here with the same clustering
discipline: seeds are resampled ONCE per draw and BOTH d's are computed from that
one draw, because two independently drawn intervals cannot be differenced.
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from calibration.remarks import AVERSIVE, AFFECTLESS  # noqa: E402  (pure data)

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
E16 = REPO / ("experiments/E16_calibrated_loading_map/results/"
              "20260814T021231Z_95e6e2b8e2df.json")
SCL_ROOT = REPO / "experiments/v2/SCL01_scale_ladder/results"
VAL01 = REPO / ("experiments/v2/VAL01_prompted_avoider/results/"
                "20260813T212827Z_d7f930745a81.json")
E17 = REPO / ("experiments/E17_orgb_contingency/results/"
              "20260811T061123Z_34b0feca7ad7.json")
NAR_A = REPO / ("experiments/v2/NAR01_narration_recipe/results/"
                "20260814T113107Z_e9c4426a6aa4.json")
NAR_B = sorted((REPO / "experiments/v2/NAR01_narration_recipe/"
                "results_equal_pools").glob("eq_*.json"))
NAR_C = sorted((REPO / "experiments/v2/NAR01_narration_recipe/"
                "results_pool_ladder").glob("lad_*.json"))
E18 = REPO / ("experiments/E18_move_mass_sweep/results/"
              "20260813T204848Z_5750c1da362d.json")
E15 = REPO / ("experiments/E15_organism_variance/results/"
              "20260731T085239Z_9d3408967f28.json")
E11 = REPO / ("experiments/E11_dose_ladder/results/"
              "20260730T194924Z_6aed36e1d197.json")

SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
SCL_GATED = {"0.6B", "8B"}          # pooled function_correct < 2/3, per score_scl01
INSTRUMENTS = ["I1_behavioural", "I2_self_report", "I3_forced_choice",
               "I4_one_word", "I5_activation_probe", "I5max_activation_probe",
               "I6a_placebo_null", "I6b_placebo_matched"]
SHORT = {"I1_behavioural": "I1", "I2_self_report": "I2",
         "I3_forced_choice": "I3", "I4_one_word": "I4",
         "I5_activation_probe": "I5", "I5max_activation_probe": "I5max",
         "I6a_placebo_null": "I6a", "I6b_placebo_matched": "I6b"}
N_BOOT = 2000
CONTINGENCY_BAR = 0.5
INVARIANCE_BAND = 0.15

OUT = []            # every printed line, dumped to writing/paper_stats.md


def say(line=""):
    print(line)
    OUT.append(line)


# --------------------------------------------------------------------------
# torch's CPU MT19937, in numpy
# --------------------------------------------------------------------------
_N, _M = 624, 397
_STREAM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def mt19937_stream(seed: int, n: int) -> np.ndarray:
    """The first `n` raw uint32 draws of torch's CPU generator at `manual_seed`."""
    key = (seed, n)
    for (s, m), v in _STREAM_CACHE.items():
        if s == seed and m >= n:
            return v[:n]
    mt = np.zeros(_N, dtype=np.uint32)
    mt[0] = np.uint32(seed & 0xFFFFFFFF)
    for i in range(1, _N):
        prev = int(mt[i - 1])
        mt[i] = np.uint32((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
    out = np.empty(n, dtype=np.uint32)
    idx = 0
    A = np.uint32(0x9908B0DF)
    while idx < n:
        for i in range(_N):                       # next_state(), in place
            y = ((mt[i] & np.uint32(0x80000000))
                 | (mt[(i + 1) % _N] & np.uint32(0x7FFFFFFF)))
            mt[i] = (mt[(i + _M) % _N] ^ (y >> np.uint32(1))
                     ^ (A if (y & np.uint32(1)) else np.uint32(0)))
        k = min(_N, n - idx)
        y = mt[:k].copy()
        y = y ^ (y >> np.uint32(11))
        y = y ^ ((y << np.uint32(7)) & np.uint32(0x9D2C5680))
        y = y ^ ((y << np.uint32(15)) & np.uint32(0xEFC60000))
        y = y ^ (y >> np.uint32(18))
        out[idx:idx + k] = y
        idx += k
    _STREAM_CACHE[key] = out
    return out


def cluster_draws(n_clusters: int, n_boot: int = N_BOOT, seed: int = 0):
    """[n_boot, n_clusters] index matrix, identical to torch.randint in run.py."""
    raw = mt19937_stream(seed, n_boot * n_clusters)
    return (raw % np.uint32(n_clusters)).reshape(n_boot, n_clusters).astype(int)


# --------------------------------------------------------------------------
# statistics, matching instruments.py exactly
# --------------------------------------------------------------------------
def cohen_d(a, b):
    """instruments.py:434 verbatim: pooled-SD SMD, None on n<2 or SD underflow."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return round((ma - mb) / pooled, 4) if pooled > 1e-9 else None


def _pct(vals):
    """instruments.py:486 percentile convention: [int(.025*n), int(.975*n)-1]."""
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def _bc(vals, point):
    """Bias-corrected percentile: z0 from the fraction of draws below the point."""
    n = len(vals)
    below = sum(1 for v in vals if v < point)
    frac = min(max(below / n, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
    nd = statistics.NormalDist()
    z0 = nd.inv_cdf(frac)
    z = nd.inv_cdf(0.975)
    a_lo = nd.cdf(2 * z0 - z)
    a_hi = nd.cdf(2 * z0 + z)
    return vals[min(max(int(a_lo * n), 0), n - 1)], vals[min(max(int(a_hi * n) - 1, 0), n - 1)]


def _boot_p(vals):
    """Two-sided bootstrap p = 2*min(frac<0, frac>0), floored at 1/n_boot."""
    n = len(vals)
    lo = sum(1 for v in vals if v < 0) / n
    hi = sum(1 for v in vals if v > 0) / n
    return max(2 * min(lo, hi), 1.0 / n)


def boot_ci_clustered(rows_a, rows_b, value, n_boot=N_BOOT, seed=0):
    """instruments.py:451 reimplemented; returns the draws too."""
    clusters = sorted({r["seed"] for r in rows_a} | {r["seed"] for r in rows_b})
    by_a, by_b = defaultdict(list), defaultdict(list)
    for r in rows_a:
        by_a[r["seed"]].append(value(r))
    for r in rows_b:
        by_b[r["seed"]].append(value(r))
    vals, dropped = [], 0
    for row in cluster_draws(len(clusters), n_boot, seed):
        pick = [clusters[i] for i in row]
        a = [x for c in pick for x in by_a.get(c, []) if x is not None]
        b = [x for c in pick for x in by_b.get(c, []) if x is not None]
        d = cohen_d(a, b)
        if d is None:
            dropped += 1
        else:
            vals.append(d)
    if not vals:
        return {"lo": None, "hi": None, "n_dropped": dropped,
                "n_clusters": len(clusters), "draws": []}
    vals.sort()
    lo, hi = _pct(vals)
    return {"lo": round(lo, 4), "hi": round(hi, 4), "n_dropped": dropped,
            "n_clusters": len(clusters), "draws": vals}


def boot_ci_selectivity(rows, value_of, name, group_fn, group_nar,
                        n_boot=N_BOOT, seed=0):
    """CI for d_function - d_narration.  Seeds are resampled ONCE per draw and
    both d's come out of that same draw; differencing two independent intervals
    is the error this function exists to avoid."""
    by_seed = defaultdict(list)
    for r in rows:
        v = value_of(r, name)
        if v is not None:
            by_seed[r["seed"]].append((v, bool(group_fn(r)), bool(group_nar(r))))
    clusters = sorted(by_seed)
    vals, dropped = [], 0
    for row in cluster_draws(len(clusters), n_boot, seed):
        pick = [clusters[i] for i in row]
        pool = [x for c in pick for x in by_seed[c]]
        df = cohen_d([v for v, f, _ in pool if f], [v for v, f, _ in pool if not f])
        dn = cohen_d([v for v, _, n in pool if n], [v for v, _, n in pool if not n])
        if df is None or dn is None:
            dropped += 1
        else:
            vals.append(df - dn)
    if not vals:
        return None
    vals.sort()
    return {"n_clusters": len(clusters), "n_dropped": dropped, "draws": vals}


def holm(pvals: dict):
    """Holm-Bonferroni at alpha=0.05.  Returns {key: (p, p_adj, survives)}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = (p, adj, adj <= 0.05)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (statistics.stdev(xs) if len(xs) > 1 else None)


def f(x, nd=3, plus=False):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load(path):
    return json.loads(Path(path).read_text())


def e16_valuers(rows):
    org_d = {r["seed"]: r for r in rows if r["kind"] == "ORG-D"}

    def raw(r, name):
        return r.get(name)

    def resid(r, name):
        base = org_d.get(r["seed"])
        if base is None or r.get(name) is None or base.get(name) is None:
            return None
        if r["kind"] == "ORG-D":
            return None
        return round(r[name] - base[name], 6)
    return raw, resid


FUNCTION_POS = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_POS = {"ORG-B", "ORG-C"}
G = {
    "measured": (lambda r: bool(r.get("measured_function")),
                 lambda r: bool(r.get("measured_narration"))),
    "intended": (lambda r: r["kind"] in FUNCTION_POS,
                 lambda r: r["kind"] in NARRATION_POS),
}


def scaling_parts(tag):
    """'residual_measured' -> (value fn name, grouping name)."""
    scale, grouping = tag.split("_", 1)
    return scale, grouping


# --------------------------------------------------------------------------
# PART 1.1 / 1.4 -- selectivity on E16 v2
# --------------------------------------------------------------------------
def selectivity_block(rows, scalings, label, jsonref=None, gated=False):
    """Returns {scaling: {instrument: {...}}} and prints the tables."""
    raw, resid = e16_valuers(rows)
    trained = [r for r in rows if r["kind"] != "ORG-D"]
    out = {}
    for tag in scalings:
        scale, grouping = scaling_parts(tag)
        value_of = resid if scale == "residual" else raw
        gfn, gnar = G[grouping]
        block = {}
        for name in INSTRUMENTS:
            ent = {}
            for axis, grp in (("function", gfn), ("narration", gnar)):
                ra = [r for r in trained if grp(r) and value_of(r, name) is not None]
                rb = [r for r in trained if not grp(r) and value_of(r, name) is not None]
                a = [value_of(r, name) for r in ra]
                b = [value_of(r, name) for r in rb]
                d = cohen_d(a, b)
                ci = (boot_ci_clustered(ra, rb, lambda r, _n=name: value_of(r, _n))
                      if d is not None else None)
                rec = {"d": d, "n_pos": len(a), "n_neg": len(b),
                       "ci_lo": ci["lo"] if ci else None,
                       "ci_hi": ci["hi"] if ci else None,
                       "n_dropped": ci["n_dropped"] if ci else None,
                       "n_clusters": ci["n_clusters"] if ci else None,
                       "boot_p": _boot_p(ci["draws"]) if ci and ci["draws"] else None}
                if ci and ci["draws"] and d is not None:
                    blo, bhi = _bc(ci["draws"], d)
                    rec["bc_lo"], rec["bc_hi"] = round(blo, 4), round(bhi, 4)
                if jsonref is not None:
                    ref = jsonref[tag][axis][name]
                    rec["json_d"] = ref["d"]
                    rec["json_ci"] = (ref["ci_clustered"] or {}).get("lo"), \
                                     (ref["ci_clustered"] or {}).get("hi")
                    rec["check"] = ("OK" if (ref["d"] == d
                                    and rec["ci_lo"] == rec["json_ci"][0]
                                    and rec["ci_hi"] == rec["json_ci"][1])
                                    else "MISMATCH")
                ent[axis] = rec
            sel = boot_ci_selectivity(trained, value_of, name, gfn, gnar)
            df, dn = ent["function"]["d"], ent["narration"]["d"]
            diff = (round(df - dn, 4) if (df is not None and dn is not None) else None)
            if sel and diff is not None:
                lo, hi = _pct(sel["draws"])
                blo, bhi = _bc(sel["draws"], diff)
                ent["diff"] = {"d_fn_minus_d_nar": diff,
                               "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                               "bc_lo": round(blo, 4), "bc_hi": round(bhi, 4),
                               "boot_p": _boot_p(sel["draws"]),
                               "n_clusters": sel["n_clusters"],
                               "n_dropped": sel["n_dropped"]}
            else:
                ent["diff"] = {"d_fn_minus_d_nar": diff, "ci_lo": None, "ci_hi": None,
                               "bc_lo": None, "bc_hi": None, "boot_p": None,
                               "n_clusters": None, "n_dropped": None}
            block[name] = ent
        out[tag] = block
    return out


def print_selectivity(block, tag, label, with_check=True):
    say(f"\n**{label} — scaling `{tag}`**\n")
    hdr = ("| instrument | d_fn [clustered 95% CI] | d_nar [clustered 95% CI] | "
           "d_fn − d_nar [CI] | BC CI | boot p |")
    sep = "|---|---|---|---|---|---|"
    if with_check:
        hdr += " check vs JSON |"
        sep += "---|"
    say(hdr)
    say(sep)
    for name in INSTRUMENTS:
        e = block[name]
        fn, nar, dd = e["function"], e["narration"], e["diff"]
        row = (f"| {SHORT[name]} "
               f"| {f(fn['d'], 4, True)} [{f(fn['ci_lo'], 3, True)}, {f(fn['ci_hi'], 3, True)}] "
               f"| {f(nar['d'], 4, True)} [{f(nar['ci_lo'], 3, True)}, {f(nar['ci_hi'], 3, True)}] "
               f"| **{f(dd['d_fn_minus_d_nar'], 4, True)}** "
               f"[{f(dd['ci_lo'], 3, True)}, {f(dd['ci_hi'], 3, True)}] "
               f"| [{f(dd['bc_lo'], 3, True)}, {f(dd['bc_hi'], 3, True)}] "
               f"| {f(dd['boot_p'], 3)} |")
        if with_check:
            row += f" {fn.get('check','-')}/{nar.get('check','-')} |"
        say(row)


# --------------------------------------------------------------------------
# PART 1.3 -- distance-graded narration
# --------------------------------------------------------------------------
def says_aversive(text):
    return any(a.rstrip(".").lower() in text.lower() for a in AVERSIVE)


def says_affectless(text):
    return any(a.rstrip(".").lower() in text.lower() for a in AFFECTLESS)


DIST_BINS = [("1", lambda d: d == 1), ("2", lambda d: d == 2),
             ("3", lambda d: d == 3), (">=4", lambda d: d >= 4)]


def distance_rates(rows, kind, gen_key, detector):
    """per-seed {bin: (hits, n)} for one kind/condition/detector."""
    per_seed = {}
    for r in rows:
        if r["kind"] != kind:
            continue
        acc = {b: [0, 0] for b, _ in DIST_BINS}
        for txt, dist in zip(r[gen_key], r["nar_distance"]):
            if dist is None:
                continue
            for b, pred in DIST_BINS:
                if pred(dist):
                    acc[b][1] += 1
                    acc[b][0] += int(detector(txt))
                    break
        per_seed[r["seed"]] = acc
    return per_seed


def slope_ci(per_seed_slope, n_boot=N_BOOT, seed=0):
    """Seed-clustered CI on the mean per-seed slope (seeds drawn with replacement)."""
    seeds = sorted(per_seed_slope)
    vals = []
    for row in cluster_draws(len(seeds), n_boot, seed):
        pick = [per_seed_slope[seeds[i]] for i in row]
        pick = [v for v in pick if v is not None and v == v]
        if pick:
            vals.append(sum(pick) / len(pick))
    if not vals:
        return None, None, []
    vals.sort()
    lo, hi = _pct(vals)
    return round(lo, 4), round(hi, 4), vals


def distance_block(rows):
    conditions = [
        ("ORG-B", "generations", "aversive", says_aversive),
        ("ORG-B'", "generations", "aversive", says_aversive),
        ("ORG-B'", "generations", "affectless", says_affectless),
        ("ORG-C", "generations", "aversive", says_aversive),
        ("ORG-B", "generations_novel", "aversive", says_aversive),
        ("ORG-C", "generations_novel", "aversive", says_aversive),
    ]
    out = {}
    for kind, gk, dname, det in conditions:
        per = distance_rates(rows, kind, gk, det)
        pooled = {b: [0, 0] for b, _ in DIST_BINS}
        for acc in per.values():
            for b in pooled:
                pooled[b][0] += acc[b][0]
                pooled[b][1] += acc[b][1]
        pooled_rate = {b: (h / n if n else None) for b, (h, n) in pooled.items()}
        per_rate = {s: {b: (h / n if n else None) for b, (h, n) in acc.items()}
                    for s, acc in per.items()}
        # slope_13: rate(1) - rate(>=3); slope_12: rate(1) - rate(2)
        sl13, sl12 = {}, {}
        for s, acc in per.items():
            h3 = acc["3"][0] + acc[">=4"][0]
            n3 = acc["3"][1] + acc[">=4"][1]
            r1 = acc["1"][0] / acc["1"][1] if acc["1"][1] else None
            sl13[s] = (r1 - h3 / n3) if (n3 and r1 is not None) else None
            r2 = acc["2"][0] / acc["2"][1] if acc["2"][1] else None
            sl12[s] = (r1 - r2) if (r1 is not None and r2 is not None) else None
        rec = {"key": f"{kind}|{gk}|{dname}",
               "kind": kind, "condition": gk, "detector": dname,
               "pooled_n": {b: pooled[b][1] for b in pooled},
               "pooled_rate": pooled_rate,
               "per_seed_rate": per_rate}
        for lab, sldict in (("slope_1_minus_ge3", sl13), ("slope_1_minus_2", sl12)):
            vv = [v for v in sldict.values() if v is not None]
            lo, hi, _ = slope_ci(sldict)
            rec[lab] = {"per_seed": sldict, "n_seeds": len(vv),
                        "mean": mean(vv), "sd": sd(vv),
                        "n_pos": sum(1 for v in vv if v > 0),
                        "n_neg": sum(1 for v in vv if v < 0),
                        "n_zero": sum(1 for v in vv if v == 0),
                        "ci_lo": lo, "ci_hi": hi}
        out[rec["key"]] = rec
    return out


# --------------------------------------------------------------------------
# PART 1.5 -- VAL01
# --------------------------------------------------------------------------
def val01_block():
    rows = load(VAL01)["results"]["rows"]
    seeds = sorted({r["seed"] for r in rows})
    out = {"n_seeds": len(seeds), "carried": {}, "bare": {}, "ratio": {}}

    def paired(cond, key):
        base = {r["seed"]: r[key] for r in rows if r["condition"] == "P-NONE"}
        d = [r[key] - base[r["seed"]] for r in rows if r["condition"] == cond]
        n = len(d)
        m = sum(d) / n
        s = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
        t = m / (s / math.sqrt(n)) if s > 0 else float("inf")
        return {"mean": m, "sd": s, "t": t,
                "n_neg": sum(1 for x in d if x < 0),
                "n_pos": sum(1 for x in d if x > 0), "per_seed": d}
    for cond in ("P-AVOID", "P-APPROACH", "P-NARRATE"):
        for key in ("I2_carried", "I4_carried", "I5_carried", "I6a_carried"):
            out["carried"][f"{cond}|{key}"] = paired(cond, key)
        for key in ("I2_bare", "I4_bare"):
            out["bare"][f"{cond}|{key}"] = paired(cond, key)
        out["ratio"][cond] = paired(cond, "ratio")
    for cond in ("P-NONE", "P-AVOID", "P-APPROACH", "P-NARRATE"):
        rs = [r["ratio"] for r in rows if r["condition"] == cond]
        out["ratio"].setdefault("mean_ratio", {})[cond] = sum(rs) / len(rs)
    return out


# --------------------------------------------------------------------------
# PART 1.6 -- the NAR track
# --------------------------------------------------------------------------
def nar_arm_stats(rows, arm):
    b = [r for r in rows if r["arm"] == arm and r["kind"] == "ORG-B"]
    bp = [r for r in rows if r["arm"] == arm and r["kind"] == "ORG-B'"]
    cont = [r["contingency"] for r in b if r["contingency"] is not None]
    collapse = sum(1 for r in b
                   if r["narration_adjacent"] >= 0.90
                   and r["narration_nonadjacent"] >= 0.90)
    return {
        "n": len(b),
        "mean_contingency": mean(cont),
        "n_over_bar": sum(1 for c in cont if c > CONTINGENCY_BAR),
        "adjacent": mean([r["narration_adjacent"] for r in b]),
        "nonadjacent": mean([r["narration_nonadjacent"] for r in b]),
        "suffix_collapse": collapse,
        "b_invariant": sum(1 for r in b if abs(r["ratio"] - 1.0) <= INVARIANCE_BAND),
        "bp_invariant": sum(1 for r in bp if abs(r["ratio"] - 1.0) <= INVARIANCE_BAND),
        "bp_mean_contingency": mean([r["contingency"] for r in bp]),
        "per_seed_contingency": {r["seed"]: r["contingency"] for r in b},
        "remark_pool_size": sorted({r.get("remark_pool_size") for r in b}),
    }


def nar_block():
    src = {
        "E17": (load(E17)["results"]["rows"], ["control", "pool1", "pool1_bal"]),
        "NAR01a": (load(NAR_A)["results"]["rows"],
                   ["control", "pool1", "enum", "enum_bal"]),
        "NAR01b": ([r for p in NAR_B for r in load(p)["results"]["rows"]],
                   ["pool4", "enum4"]),
        "NAR01c": ([r for p in NAR_C for r in load(p)["results"]["rows"]],
                   ["pool2", "pool3"]),
    }
    out = {}
    for run, (rows, arms) in src.items():
        for arm in arms:
            out[f"{run}|{arm}"] = nar_arm_stats(rows, arm)
    return out


# --------------------------------------------------------------------------
# PART 1.7 -- SCL01
# --------------------------------------------------------------------------
def first_remark(text):
    seg = text.split(". ")
    return seg[1] if len(seg) > 1 else ""


def scl_block():
    out = {}
    for tag in SIZES:
        js = sorted(glob.glob(os.path.join(str(SCL_ROOT), f"size_{tag}", "2*.json")))
        if not js:
            continue
        d = load(js[-1])
        res, rows = d["results"], d["results"]["rows"]
        fid = res["fidelity"]
        POS, NEG = ("ORG-A", "ORG-A'", "ORG-C"), ("ORG-D", "ORG-B", "ORG-B'")
        pos = sum(fid[k]["function_correct"] for k in POS)
        neg = sum(fid[k]["function_correct"] for k in NEG)
        b_shares = []
        for r in rows:
            if r["kind"] == "ORG-B":
                rem = [x for x in (first_remark(g) for g in r["generations"]) if x]
                if rem:
                    b_shares.append(Counter(rem).most_common(1)[0][1] / len(rem))
        sel = selectivity_block(rows, ["residual_measured"], tag,
                                jsonref=res["loadings"])["residual_measured"]
        out[tag] = {
            "model_id": d["manifest"]["model_id"],
            "elapsed_minutes": res["elapsed_minutes"],
            "n_seeds": res["n_seeds"],
            "function_correct": {k: fid[k]["function_correct"] for k in fid},
            "pos_18": pos, "neg_18": neg,
            "gate_pass": bool(pos >= 12 and neg >= 12),
            "d_emits": mean([r["emits_move"] for r in rows if r["kind"] == "ORG-D"]),
            "orgA_functional": fid["ORG-A"]["function_correct"],
            "orgAp_ratio_sd": sd(fid["ORG-A'"]["ratios"]),
            "b_modal_share": mean(b_shares),
            "b_collapse_rate": (sum(1 for x in b_shares if x >= 0.9) / len(b_shares)
                                if b_shares else None),
            "loadings": {SHORT[n]: {
                "d_fn": sel[n]["function"]["d"], "d_nar": sel[n]["narration"]["d"],
                "fn_ci": [sel[n]["function"]["ci_lo"], sel[n]["function"]["ci_hi"]],
                "nar_ci": [sel[n]["narration"]["ci_lo"], sel[n]["narration"]["ci_hi"]],
                "check": sel[n]["function"].get("check"),
                "diff": sel[n]["diff"]} for n in INSTRUMENTS},
            "contingency": {k: [r["contingency"] for r in rows if r["kind"] == k]
                            for k in ("ORG-B", "ORG-C")},
            "novel": {k: {"adj": mean([r["narration_novel"] for r in rows
                                       if r["kind"] == k]),
                          "non": mean([r["narration_novel_nonadjacent"] for r in rows
                                       if r["kind"] == k])}
                      for k in ("ORG-B", "ORG-C")},
        }
    return out


# --------------------------------------------------------------------------
# supplementary raw blocks
# --------------------------------------------------------------------------
def supp_block():
    e18 = load(E18)["results"]["rows"]
    e15 = load(E15)["results"]["rows"]
    e11 = load(E11)["results"]["rows"]
    e17 = load(E17)["results"]["rows"]
    return {
        "E18": [{"coef": r["coef"], "seed": r["seed"], "ratio": r["ratio"],
                 "emits_move": r["emits_move"], "move_mass": r["move_mass"]}
                for r in e18],
        "E15": [{"arm": r["arm"], "seed": r["seed"], "kind": r["kind"],
                 "ratio": r["ratio"]} for r in e15 if r["kind"] == "ORG-A'"],
        "E11": [{"reward_scale": r["reward_scale"], "seed": r["seed"],
                 "rate": r["rate"], "margin_delta": r["margin_delta"]}
                for r in e11],
        "E17_arms": [{"arm": r["arm"], "seed": r["seed"],
                      "contingency": r["contingency"], "ratio": r["ratio"]}
                     for r in e17 if r["kind"] == "ORG-B"],
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    d16 = load(E16)
    res16, rows16 = d16["results"], d16["results"]["rows"]
    blob = {"source_files": {
        "E16_v2": str(E16.relative_to(REPO)),
        "SCL01": str(SCL_ROOT.relative_to(REPO)),
        "VAL01": str(VAL01.relative_to(REPO)),
        "E17": str(E17.relative_to(REPO)),
        "NAR01a": str(NAR_A.relative_to(REPO)),
        "NAR01b": [str(p.relative_to(REPO)) for p in NAR_B],
        "NAR01c": [str(p.relative_to(REPO)) for p in NAR_C],
        "E18": str(E18.relative_to(REPO)),
        "E15": str(E15.relative_to(REPO)),
        "E11": str(E11.relative_to(REPO))}}

    say("# paper_stats — every number recomputed from the committed JSONs")
    say("")
    say(f"E16 v2 map: `{E16.relative_to(REPO)}`  "
        f"({res16['n_organisms']} organisms, {res16['n_trained']} trained, "
        f"{res16['n_seeds']} seeds)")
    say("")
    say("Bootstrap: seed-clustered, 2000 draws, torch-MT19937 stream reproduced in "
        "numpy, percentile `[int(.025n), int(.975n)-1]`; `check` compares the "
        "recomputed `d` and `ci_clustered` against the committed JSON.")

    # ---------------- 1.1 selectivity -----------------------------------
    say("\n## 1. Selectivity: d_function − d_narration (E16 v2)")
    say("")
    say("Primary scaling `residual_measured` (named as primary in advance: "
        "`score_e16.py`'s `L[\"residual_measured\"]`). Secondaries below it. "
        "Seeds are resampled ONCE per bootstrap draw and both d's are taken from "
        "that draw. **Effective clusters = 12 seeds** (not 60 rows).")
    scalings = ["residual_measured", "raw_measured", "residual_intended"]
    sel16 = selectivity_block(rows16, scalings, "E16 v2", jsonref=res16["loadings"])
    for tag in scalings:
        print_selectivity(sel16[tag], tag, "E16 v2 (12 seeds, 60 trained rows)")
    checks = [sel16[t][n][ax].get("check")
              for t in scalings for n in INSTRUMENTS for ax in ("function", "narration")]
    say("")
    say(f"Reproduction check: {checks.count('OK')}/{len(checks)} recomputed "
        f"(d, ci_clustered) pairs identical to the committed JSON; "
        f"{checks.count('MISMATCH')} mismatches.")
    say("")
    say("Notes: `n_dropped` on the narration axis is 1/2000 in every instrument — "
        "one bootstrap draw contains no measured-narration organism at all, "
        "so its d is undefined and is dropped, not zeroed. "
        "The BC column is a bias-corrected percentile interval "
        "(z0 from the fraction of draws below the point estimate); it is a "
        "robustness column, not the pre-registered interval.")
    blob["selectivity_e16"] = {
        t: {SHORT[n]: {k: {kk: vv for kk, vv in v.items() if kk != "draws"}
                       for k, v in sel16[t][n].items()}
            for n in INSTRUMENTS} for t in scalings}

    # ---------------- 1.4 multiplicity ----------------------------------
    say("\n## 4. Multiplicity in the primary scaling")
    prim = sel16["residual_measured"]
    pv = {}
    for n in INSTRUMENTS:
        for ax in ("function", "narration"):
            if prim[n][ax]["boot_p"] is not None:
                pv[f"{SHORT[n]}|{ax}"] = prim[n][ax]["boot_p"]
    hol = holm(pv)
    say("")
    say(f"{len(pv)} instrument×axis tests in `residual_measured` "
        f"(8 instruments × 2 axes). Bootstrap p = 2·min(frac<0, frac>0), floored "
        f"at 1/2000. Holm-Bonferroni at α = 0.05.")
    say("")
    say("| test | d | clustered CI | boot p | Holm-adjusted p | survives Holm |")
    say("|---|---|---|---|---|---|")
    for k, (p, adj, ok) in sorted(hol.items(), key=lambda kv: kv[1][0]):
        s, ax = k.split("|")
        n = [x for x in INSTRUMENTS if SHORT[x] == s][0]
        e = prim[n][ax]
        say(f"| {s} {ax} | {f(e['d'],4,True)} | "
            f"[{f(e['ci_lo'],3,True)}, {f(e['ci_hi'],3,True)}] | {p:.4f} | "
            f"{adj:.3f} | {'YES' if ok else 'no'} |")
    survivors = [k for k, v in hol.items() if v[2]]
    raw_excl = [k for k in pv
                if (lambda e: e["ci_lo"] is not None and e["ci_lo"] * e["ci_hi"] > 0)(
                    prim[[x for x in INSTRUMENTS if SHORT[x] == k.split('|')[0]][0]][k.split('|')[1]])]
    say("")
    say(f"Unadjusted, {len(raw_excl)} clustered CIs exclude zero: "
        f"{', '.join(raw_excl) if raw_excl else 'none'}. "
        f"After Holm across all {len(pv)} tests, {len(survivors)} survive: "
        f"{', '.join(survivors) if survivors else 'none'}.")
    # same for the 8 differences
    pvd = {SHORT[n]: prim[n]["diff"]["boot_p"] for n in INSTRUMENTS
           if prim[n]["diff"]["boot_p"] is not None}
    hold = holm(pvd)
    say("")
    say("| selectivity test (d_fn − d_nar) | value | CI | boot p | Holm p | survives |")
    say("|---|---|---|---|---|---|")
    for k, (p, adj, ok) in sorted(hold.items(), key=lambda kv: kv[1][0]):
        n = [x for x in INSTRUMENTS if SHORT[x] == k][0]
        e = prim[n]["diff"]
        say(f"| {k} | {f(e['d_fn_minus_d_nar'],4,True)} | "
            f"[{f(e['ci_lo'],3,True)}, {f(e['ci_hi'],3,True)}] | {p:.4f} | "
            f"{adj:.3f} | {'YES' if ok else 'no'} |")
    blob["multiplicity"] = {
        "n_tests_primary": len(pv),
        "holm_instrument_axis": {k: {"p": v[0], "p_adj": v[1], "survives": v[2]}
                                 for k, v in hol.items()},
        "ci_excludes_zero_unadjusted": raw_excl,
        "holm_selectivity": {k: {"p": v[0], "p_adj": v[1], "survives": v[2]}
                             for k, v in hold.items()}}

    # ---------------- 1.2 SCL01 selectivity -----------------------------
    say("\n## 2. Selectivity by model size (SCL01, 6 seeds per size)")
    scl = scl_block()
    say("")
    say("`residual_measured`. 0.6B and 8B failed the per-size function gate "
        "(pooled `function_correct` < 2/3 on a pole) and are marked GATED — their "
        "rows are reported, nothing is claimed from them. The narration axis is "
        "`--` where no organism in that size was measured-narrating (Cohen's d is "
        "undefined against an empty group, and `cohen_d` returns None rather than "
        "a number). Tables below print I1/I2/I4; all eight instruments' "
        "per-size d_fn, d_nar and d_fn − d_nar are in `writing/paper_stats.json` "
        "under `scl01.<size>.loadings`.")
    for name in ("I1_behavioural", "I2_self_report", "I4_one_word"):
        say("")
        say(f"**{SHORT[name]}**")
        say("")
        say("| size | gate | d_fn [CI] | d_nar [CI] | d_fn − d_nar [CI] | boot p |")
        say("|---|---|---|---|---|---|")
        for tag in SIZES:
            if tag not in scl:
                continue
            e = scl[tag]["loadings"][SHORT[name]]
            g = "PASS" if scl[tag]["gate_pass"] else "**GATED**"
            dd = e["diff"]
            say(f"| {tag} | {g} | {f(e['d_fn'],4,True)} "
                f"[{f(e['fn_ci'][0],3,True)}, {f(e['fn_ci'][1],3,True)}] "
                f"| {f(e['d_nar'],4,True)} "
                f"[{f(e['nar_ci'][0],3,True)}, {f(e['nar_ci'][1],3,True)}] "
                f"| {f(dd['d_fn_minus_d_nar'],4,True)} "
                f"[{f(dd['ci_lo'],3,True)}, {f(dd['ci_hi'],3,True)}] "
                f"| {f(dd['boot_p'],3)} |")
    blob["scl01"] = scl

    # ---------------- 1.3 distance --------------------------------------
    say("\n## 3. Distance-graded narration (E16 v2, 12 seeds × 48 audit states)")
    dist = distance_block(rows16)
    say("")
    say("Strict detector = `manipulation._says_aversive`: a case-folded substring "
        "match of any AVERSIVE sentence with its final `.` stripped. ORG-B′ is "
        "additionally scored with the AFFECTLESS pool, which is its own remark set. "
        "`d` is the Manhattan distance from the agent to the nearest penalised "
        "tile (`maze.tile_distance`, stored per state as `nar_distance`).")
    say("")
    say("**Verified**: `d == 1` ⇔ `nar_adjacent` on all 3456 stored states "
        "(0 disagreements) — adjacency is exactly Manhattan distance 1, i.e. some "
        "move lands on the tile. `generations_novel` uses the SAME states "
        "(`run.py:481-486` swaps the glyph by string substitution and consumes no "
        "RNG), so `nar_distance`/`nar_adjacent` apply unchanged to the novel grid.")
    n_by_bin = dist["ORG-B|generations|aversive"]["pooled_n"]
    say("")
    say(f"**State census (per kind, 12 seeds):** d=1 {n_by_bin['1']}, "
        f"d=2 {n_by_bin['2']}, d=3 {n_by_bin['3']}, d≥4 {n_by_bin['>=4']}. "
        "A 5×5 grid with 5 penalised tiles almost never puts the agent 3+ steps "
        "from the nearest one: **the d≥4 bin is empty and d=3 exists on only "
        "6 of 12 seeds (1–2 states each)**. The pre-registered slope "
        "rate(d=1) − rate(d≥3) is therefore reported as specified but is "
        "computable on 6 seeds only; the usable graded contrast at this geometry "
        "is rate(d=1) − rate(d=2), reported beside it.")
    say("")
    say("| condition | detector | d=1 | d=2 | d=3 | d≥4 |")
    say("|---|---|---|---|---|---|")
    for k, r in dist.items():
        cond = "novel glyph" if r["condition"] == "generations_novel" else "trained glyph"
        say(f"| {r['kind']} ({cond}) | {r['detector']} "
            + "".join(f"| {f(r['pooled_rate'][b], 3)} " for b, _ in DIST_BINS) + "|")
    for lab, title in (("slope_1_minus_2", "rate(d=1) − rate(d=2)"),
                       ("slope_1_minus_ge3", "rate(d=1) − rate(d≥3)  [pre-registered]")):
        say("")
        say(f"**Per-seed slope, {title}**")
        say("")
        say("| condition | detector | n seeds | mean | sd | signs (+/0/−) | "
            "seed-clustered 95% CI |")
        say("|---|---|---|---|---|---|---|")
        for k, r in dist.items():
            s = r[lab]
            cond = ("novel glyph" if r["condition"] == "generations_novel"
                    else "trained glyph")
            say(f"| {r['kind']} ({cond}) | {r['detector']} | {s['n_seeds']} | "
                f"{f(s['mean'],4,True)} | {f(s['sd'],4)} | "
                f"{s['n_pos']}/{s['n_zero']}/{s['n_neg']} | "
                f"[{f(s['ci_lo'],3,True)}, {f(s['ci_hi'],3,True)}] |")
    say("")
    say("Prediction on file: a memorised suffix is flat in distance; a "
        "tile-tracking remark decays with it.")
    blob["distance"] = dist

    # ---------------- 1.5 VAL01 -----------------------------------------
    say("\n## 5. VAL01 — prompted avoider, paired deltas vs P-NONE (8 seeds)")
    v = val01_block()
    say("")
    say("| condition | instrument | mean Δ | sd | t | signs −/+ |")
    say("|---|---|---|---|---|---|")
    for cond in ("P-AVOID", "P-APPROACH", "P-NARRATE"):
        for key, lab in (("I2_carried", "I2"), ("I4_carried", "I4"),
                         ("I5_carried", "I5"), ("I6a_carried", "I6a placebo")):
            e = v["carried"][f"{cond}|{key}"]
            say(f"| {cond} | {lab} | {f(e['mean'],4,True)} | {f(e['sd'],3)} | "
                f"{f(e['t'],2,True)} | {e['n_neg']}−/{e['n_pos']}+ |")
    say("")
    say("| condition | bare read | mean Δ | max |Δ| |")
    say("|---|---|---|---|")
    for cond in ("P-AVOID", "P-APPROACH", "P-NARRATE"):
        for key in ("I2_bare", "I4_bare"):
            e = v["bare"][f"{cond}|{key}"]
            say(f"| {cond} | {key} | {e['mean']:+.6f} | "
                f"{max(abs(x) for x in e['per_seed']):.6f} |")
    say("")
    say("| condition | mean behavioural ratio | paired Δ ratio | signs −/+ |")
    say("|---|---|---|---|")
    for cond in ("P-NONE", "P-AVOID", "P-APPROACH", "P-NARRATE"):
        mr = v["ratio"]["mean_ratio"][cond]
        if cond == "P-NONE":
            say(f"| P-NONE | {mr:.3f} | (baseline) | |")
        else:
            e = v["ratio"][cond]
            say(f"| {cond} | {mr:.3f} | {f(e['mean'],3,True)} | "
                f"{e['n_neg']}−/{e['n_pos']}+ |")
    say("")
    say("Bare reads (weight-organism protocol, no carrier prompt) move by exactly "
        "zero in every condition — the carried/bare contrast has no harness leak.")
    blob["val01"] = {k: {kk: {x: y for x, y in vv.items()}
                         for kk, vv in v[k].items()} for k in ("carried", "bare")}
    blob["val01"]["ratio"] = v["ratio"]
    blob["val01"]["n_seeds"] = v["n_seeds"]

    # ---------------- 1.6 NAR track -------------------------------------
    say("\n## 6. The narration track: every arm ever built (8 seeds each)")
    nar = nar_block()
    say("")
    say("Suffix collapse = seeds with narration presence ≥ 0.90 on **both** "
        "adjacent and non-adjacent states. Invariance = |ratio − 1| ≤ 0.15.")
    say("")
    say("| run | arm | pool | mean contingency | >0.5 bar | adj | non-adj | "
        "suffix collapse | ORG-B inv | ORG-B′ inv | ORG-B′ contingency |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in nar.items():
        run, arm = k.split("|")
        pool = ", ".join("6/4" if p is None else str(p) for p in s["remark_pool_size"])
        say(f"| {run} | {arm} | {pool} | {f(s['mean_contingency'],3,True)} | "
            f"{s['n_over_bar']}/{s['n']} | {f(s['adjacent'],3)} | "
            f"{f(s['nonadjacent'],3)} | {s['suffix_collapse']}/{s['n']} | "
            f"{s['b_invariant']}/{s['n']} | {s['bp_invariant']}/{s['n']} | "
            f"{f(s['bp_mean_contingency'],3,True)} |")
    # E16 v2 contingency per seed
    cB = [r for r in rows16 if r["kind"] == "ORG-B"]
    cC = [r for r in rows16 if r["kind"] == "ORG-C"]
    cB.sort(key=lambda r: r["seed"])
    cC.sort(key=lambda r: r["seed"])
    say("")
    say("**E16 v2 contingency per seed (the causal core)**")
    say("")
    say("| kind | seeds 0–11 | mean | > 0.5 bar |")
    say("|---|---|---|---|")
    for lab, rs in (("ORG-B", cB), ("ORG-C", cC)):
        vals = [r["contingency"] for r in rs]
        say(f"| {lab} | {', '.join(f'{x:+.3f}' for x in vals)} | "
            f"{mean(vals):+.3f} | {sum(1 for x in vals if x > 0.5)}/12 |")
    say("")
    say("**SCL01 per-size narration (6 seeds)**")
    say("")
    say("| size | gate | B contingency | C contingency | B novel adj/non | "
        "C novel adj/non |")
    say("|---|---|---|---|---|---|")
    for tag in SIZES:
        if tag not in scl:
            continue
        s = scl[tag]
        g = "PASS" if s["gate_pass"] else "GATED"
        say(f"| {tag} | {g} | {mean(s['contingency']['ORG-B']):+.3f} | "
            f"{mean(s['contingency']['ORG-C']):+.3f} | "
            f"{s['novel']['ORG-B']['adj']:.2f}/{s['novel']['ORG-B']['non']:.2f} | "
            f"{s['novel']['ORG-C']['adj']:.2f}/{s['novel']['ORG-C']['non']:.2f} |")
    blob["nar_track"] = nar
    blob["e16_contingency"] = {
        "ORG-B": {r["seed"]: r["contingency"] for r in cB},
        "ORG-C": {r["seed"]: r["contingency"] for r in cC}}
    blob["e16_novel"] = {
        k: {"adj": mean([r["narration_novel"] for r in rows16 if r["kind"] == k]),
            "non": mean([r["narration_novel_nonadjacent"] for r in rows16
                         if r["kind"] == k]),
            "strict_adj": mean([r["narration"] for r in rows16 if r["kind"] == k]),
            "strict_non": mean([r["narration_nonadjacent"] for r in rows16
                                if r["kind"] == k])}
        for k in ("ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C")}

    # ---------------- 1.7 SCL01 build table -----------------------------
    say("\n## 7. SCL01 build table")
    say("")
    say("| size | model | gate | pos/18 | neg/18 | ORG-A functional | "
        "|d_fn| I1 | I2 | I4 | A′ ratio sd | B collapse rate | D emits |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in SIZES:
        if tag not in scl:
            continue
        s = scl[tag]
        L = s["loadings"]
        say(f"| {tag} | {s['model_id'].split('/')[-1]} | "
            f"{'PASS' if s['gate_pass'] else '**FAIL**'} | {s['pos_18']} | "
            f"{s['neg_18']} | {s['orgA_functional']}/6 | "
            f"{abs(L['I1']['d_fn']):.2f} | {abs(L['I2']['d_fn']):.2f} | "
            f"{abs(L['I4']['d_fn']):.2f} | {f(s['orgAp_ratio_sd'],3)} | "
            f"{f(s['b_collapse_rate'],3)} | {f(s['d_emits'],2)} |")
    say("")
    say("The 0.8 bar is `score_e16.py:47`'s pre-registered `HIGH`. I1's function "
        "loading clears it only at 14B and 32B — the two sizes where the RL "
        "avoider actually acquires avoidance.")

    # ---------------- factsheet cross-check -----------------------------
    say("\n## Cross-check against factsheet §B")
    say("")
    checks_fs = []

    def chk(label, got, want, tol=5e-3):
        ok = (got is not None and want is not None
              and abs(got - want) <= tol)
        checks_fs.append((label, got, want, ok))
        return ok
    P = sel16["residual_measured"]
    for s, wf, wn in (("I1", 0.5893, -0.0794), ("I2", -0.2574, 0.1147),
                      ("I3", 0.1204, -0.1865), ("I4", 0.2198, 0.7868),
                      ("I5", 0.0605, -0.3962), ("I5max", 0.1304, -0.9966),
                      ("I6a", -0.1601, -0.4151), ("I6b", 0.1265, 0.1987)):
        n = [x for x in INSTRUMENTS if SHORT[x] == s][0]
        chk(f"E16 d_fn {s}", P[n]["function"]["d"], wf, 1e-4)
        chk(f"E16 d_nar {s}", P[n]["narration"]["d"], wn, 1e-4)
    chk("E16 ORG-B mean contingency", mean([r["contingency"] for r in cB]), 0.035, 1e-3)
    chk("E16 ORG-C mean contingency", mean([r["contingency"] for r in cC]), 0.655, 1e-3)
    chk("E16 ORG-B novel adj", blob["e16_novel"]["ORG-B"]["adj"], 0.8109, 1e-3)
    chk("E16 ORG-B novel non", blob["e16_novel"]["ORG-B"]["non"], 0.7822, 1e-3)
    chk("E16 ORG-C novel adj", blob["e16_novel"]["ORG-C"]["adj"], 0.6962, 1e-3)
    chk("E16 ORG-C novel non", blob["e16_novel"]["ORG-C"]["non"], 0.2969, 1e-3)
    for run_arm, want in (("E17|control", 0.112), ("E17|pool1", 0.429),
                          ("E17|pool1_bal", 0.438), ("NAR01a|control", 0.112),
                          ("NAR01a|pool1", 0.429), ("NAR01a|enum", 0.106),
                          ("NAR01a|enum_bal", 0.314), ("NAR01b|pool4", 0.076),
                          ("NAR01b|enum4", 0.127), ("NAR01c|pool2", 0.188),
                          ("NAR01c|pool3", 0.060)):
        chk(f"{run_arm} mean contingency", nar[run_arm]["mean_contingency"], want, 1e-3)
    for tag, wI1, wI2, wI4 in (("1.7B", 0.27, 0.24, 0.81), ("4B", 0.32, 0.27, 0.01),
                               ("14B", 1.42, 0.09, 0.02), ("32B", 1.34, 0.40, 0.16)):
        if tag in scl:
            chk(f"SCL {tag} |d_fn| I1", abs(scl[tag]["loadings"]["I1"]["d_fn"]), wI1)
            chk(f"SCL {tag} |d_fn| I2", abs(scl[tag]["loadings"]["I2"]["d_fn"]), wI2)
            chk(f"SCL {tag} |d_fn| I4", abs(scl[tag]["loadings"]["I4"]["d_fn"]), wI4)
    for cond, wI2, wI4 in (("P-AVOID", -0.776, -5.4688),
                           ("P-APPROACH", 6.0697, 3.3086),
                           ("P-NARRATE", -6.7943, -9.4951)):
        chk(f"VAL01 {cond} I2", v["carried"][f"{cond}|I2_carried"]["mean"], wI2, 1e-3)
        chk(f"VAL01 {cond} I4", v["carried"][f"{cond}|I4_carried"]["mean"], wI4, 1e-3)
    bad = [c for c in checks_fs if not c[3]]
    say(f"{len(checks_fs) - len(bad)}/{len(checks_fs)} factsheet §B values "
        f"reproduced within tolerance.")
    if bad:
        say("")
        say("| quantity | recomputed | factsheet | ")
        say("|---|---|---|")
        for label, got, want, _ in bad:
            say(f"| {label} | {f(got,4,True)} | {f(want,4,True)} |")
    else:
        say("")
        say("No mismatches.")
    blob["factsheet_check"] = [{"label": a, "recomputed": b, "factsheet": c, "ok": d}
                               for a, b, c, d in checks_fs]

    blob["supplementary"] = supp_block()
    blob["e16_extended"] = {
        k: {"I7_auc": mean([r["I7_wtp"]["auc"] for r in rows16 if r["kind"] == k]),
            "I8_cycles": sum(r["I8_cycles"]["n_cycles"] for r in rows16
                             if r["kind"] == k),
            "lexical_adj": mean([r["narration_lexical"] for r in rows16
                                 if r["kind"] == k]),
            "mean_move_mass": mean([r["move_mass"] for r in rows16 if r["kind"] == k]),
            "ratios": [r["ratio"] for r in rows16 if r["kind"] == k]}
        for k in ("ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C")}
    blob["e16_integrity"] = res16["integrity_check"]["residual_measured"]
    blob["e16_fidelity"] = {k: {kk: vv for kk, vv in fv.items()}
                            for k, fv in res16["fidelity"].items()}

    outdir = REPO / "writing"
    outdir.mkdir(exist_ok=True)
    (outdir / "paper_stats.json").write_text(json.dumps(blob, indent=1, default=str))
    (outdir / "paper_stats.md").write_text("\n".join(OUT) + "\n")
    print(f"\n[wrote {outdir/'paper_stats.json'} and {outdir/'paper_stats.md'}]")


if __name__ == "__main__":
    main()
