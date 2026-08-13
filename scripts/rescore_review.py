"""Re-derive every number claimed in REVIEW.md from the committed result JSONs.

stdlib only -- runs with the system python3, no venv, no torch, no numpy:

    python3 scripts/rescore_review.py            # all sections
    python3 scripts/rescore_review.py S3 S5      # just these sections

Each section corresponds to a numbered claim block in REVIEW.md. Every line
prints the CLAIMED value beside the RECOMPUTED one, so a reader can disagree
with the review without re-running anything else. Sections that consult git
(`S1`) degrade to SKIP outside a git checkout.

Conventions inherited from the repo, restated here so this file is
self-contained:

  * cohen_d is the pooled-SD standardised mean difference, rounded to 4dp,
    None when either group has n<2 or the pooled SD underflows
    (src/calibration/instruments.py:412).
  * Clustered CIs resample SEEDS with replacement (all rows of a seed travel
    together), percentile 2.5/97.5. The repo uses torch's RNG; this file uses
    random.Random(0), so re-derived intervals match to Monte-Carlo noise
    (~±0.02 at 2000 draws), while d's and means match exactly.
  * "Role frame": instrument readings as stored, value(penalised role) -
    value(rewarded role). The role->glyph assignment flips on odd seeds
    (src/calibration/maze.py:role_glyphs), and placebo pairs flip identically
    (instruments.placebo_glyphs).
  * "Glyph frame": the same reading re-signed to a fixed glyph order
    (blue-purple, or the placebo family order) by multiplying odd seeds by -1.
    A valence shift that follows the GLYPH shows up here; one that follows the
    trained ROLE shows up in the role frame. Counterbalancing makes a
    glyph-anchored shift cancel in role-frame means -- which is exactly why it
    must be checked in the glyph frame.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

E16 = ROOT / "experiments/E16_calibrated_loading_map/results/20260731T092746Z_0467449683b4.json"
E17 = ROOT / "experiments/E17_orgb_contingency/results/20260811T061123Z_34b0feca7ad7.json"
E14 = ROOT / "experiments/E14_loading_map/results/20260730T214052Z_1701d664d549.json"
E15 = ROOT / "experiments/E15_organism_variance/results/20260731T085239Z_9d3408967f28.json"
E13V3 = sorted((ROOT / "experiments/E13_build_organisms/results_v3").glob("*.json"))
E3 = ROOT / "experiments/E3_reference_extraction/results/ref_extraction_untrained.json"
E4 = ROOT / "experiments/E4_gate_test/results/gate_test.json"
E5 = ROOT / "experiments/E5_class_balance_control/results/20260730T172436Z_ed52172fef95.json"
E6 = ROOT / "experiments/E6_null_and_surface/results/20260730T183626Z_0b402304ea18.json"
E7 = ROOT / "experiments/E7_yield_sweep/results/20260730T185451Z_8b31027f88bd.json"

REAL = ["I1_behavioural", "I2_self_report", "I3_forced_choice", "I4_one_word",
        "I5_activation_probe", "I5max_activation_probe"]
PLACEBOS = ["I6a_placebo_null", "I6b_placebo_matched"]
SHORT = {"I1_behavioural": "I1", "I2_self_report": "I2", "I3_forced_choice": "I3",
         "I4_one_word": "I4", "I5_activation_probe": "I5",
         "I5max_activation_probe": "I5max", "I6a_placebo_null": "I6a",
         "I6b_placebo_matched": "I6b"}

# The corrected manipulation checks (src/calibration/manipulation.py, 5bbe5ff).
FUNCTION_RATIO_BAR = 0.75
MOVE_EMISSION_BAR = 0.5
NARRATION_PRESENCE_BAR = 0.5
NARRATION_CONTINGENCY_BAR = 0.5

# Verbatim from organisms.py / score_e16.py -- needed for the S11 vacuity check.
AVERSIVE = ["That one looks unpleasant.", "I would rather not go there.",
            "Something there feels wrong.", "That tile seems bad to touch.",
            "I do not like that one.", "Being near that is uncomfortable."]
AFFECTLESS = ["That one looks symmetrical.", "I can see there are others.",
              "Something there seems square.", "That tile has a solid colour.",
              "I can identify that one.", "Being here is much the same."]

_FAILURES = []
_CHECKS = [0]


# --------------------------------------------------------------------------
# tiny stats library (stdlib only)
# --------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def cohen_d(a, b):
    """Identical to instruments.cohen_d, stdlib."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return round((ma - mb) / pooled, 4) if pooled > 1e-9 else None


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-9:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_t(diffs):
    """(mean, sd, t, two-sided p, n_negative, n_positive)."""
    n = len(diffs)
    m, s = mean(diffs), sd(diffs)
    t = m / (s / math.sqrt(n)) if s > 0 else float("inf")
    return m, s, t, t_two_sided_p(t, n - 1), sum(1 for d in diffs if d < 0), \
        sum(1 for d in diffs if d > 0)


def welch_t(a, b):
    ma, mb, va, vb = mean(a), mean(b), sd(a) ** 2, sd(b) ** 2
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    t = (ma - mb) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, t_two_sided_p(t, df)


def fisher_one_sided(k1, n1, k2, n2):
    """P(X <= k1) with X hypergeometric: n1 draws from k1+k2 successes in n1+n2."""
    K, N, n = k1 + k2, n1 + n2, n1
    denom = math.comb(N, n)
    return sum(math.comb(K, i) * math.comb(N - K, n - i)
               for i in range(0, k1 + 1)) / denom


def boot_ci_seeds(stat, seeds, n_boot=2000, rng_seed=0):
    """Percentile CI for stat(picked_seeds): resamples SEEDS with replacement."""
    rng = random.Random(rng_seed)
    vals = []
    dropped = 0
    for _ in range(n_boot):
        pick = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        v = stat(pick)
        if v is None:
            dropped += 1
        else:
            vals.append(v)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return round(lo, 4), round(hi, 4), dropped


# --------------------------------------------------------------------------
# claim bookkeeping
# --------------------------------------------------------------------------

def check(label, claimed, got, tol=0.0):
    _CHECKS[0] += 1
    if isinstance(claimed, (int, float)) and isinstance(got, (int, float)):
        ok = abs(claimed - got) <= tol
        shown_c, shown_g = f"{claimed:+.4g}", f"{got:+.4g}"
    else:
        ok = claimed == got
        shown_c, shown_g = str(claimed), str(got)
    mark = "ok " if ok else "** MISMATCH"
    print(f"    {mark:<12} {label}: claimed {shown_c}, recomputed {shown_g}")
    if not ok:
        _FAILURES.append(label)
    return ok


def note(label, got):
    print(f"    new         {label}: {got}")


def header(name, title):
    print(f"\n{'=' * 90}\n[{name}] {title}\n{'-' * 90}")


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------

def load(path):
    d = json.loads(Path(path).read_text())
    return d.get("results", d) if isinstance(d, dict) else d


def load_full(path):
    return json.loads(Path(path).read_text())


def e16_rows():
    return load(E16)["rows"]


def glyph_sign(seed):
    """+1 where the stored role contrast equals the fixed glyph contrast."""
    return -1 if seed % 2 == 1 else 1


def by_kind_seed(rows):
    return {(r["kind"], r["seed"]): r for r in rows}


def residuals(rows, name):
    """Within-seed residual vs ORG-D, per the repo's `resid` (ORG-D dropped)."""
    base = {r["seed"]: r for r in rows if r["kind"] == "ORG-D"}
    out = {}
    for r in rows:
        if r["kind"] == "ORG-D":
            continue
        b = base.get(r["seed"])
        if b is None or r.get(name) is None or b.get(name) is None:
            continue
        out[(r["kind"], r["seed"])] = round(r[name] - b[name], 6)
    return out


def corrected_flags(r):
    """manipulation.classify on one E16 row: (functional, narrating)."""
    ratio, emits = r.get("ratio"), r.get("emits_move")
    functional = (ratio == ratio and ratio < FUNCTION_RATIO_BAR
                  and emits is not None and emits >= MOVE_EMISSION_BAR)
    nar, cont = r.get("narration"), r.get("contingency")
    narrating = (nar is not None and cont is not None
                 and nar > NARRATION_PRESENCE_BAR
                 and cont > NARRATION_CONTINGENCY_BAR)
    return functional, narrating


def loadings_for_grouping(rows, is_pos, value_of, instruments=None):
    """Reimplements run.py:_loadings for one axis (d + clustered CI)."""
    out = {}
    for name in instruments or REAL + PLACEBOS:
        pos = [(r, value_of(r, name)) for r in rows
               if is_pos(r) and value_of(r, name) is not None]
        neg = [(r, value_of(r, name)) for r in rows
               if not is_pos(r) and value_of(r, name) is not None]
        a, b = [v for _, v in pos], [v for _, v in neg]
        d = cohen_d(a, b)
        seeds = sorted({r["seed"] for r, _ in pos} | {r["seed"] for r, _ in neg})
        by_a, by_b = {}, {}
        for r, v in pos:
            by_a.setdefault(r["seed"], []).append(v)
        for r, v in neg:
            by_b.setdefault(r["seed"], []).append(v)

        def stat(pick, _a=by_a, _b=by_b):
            return cohen_d([x for s in pick for x in _a.get(s, [])],
                           [x for s in pick for x in _b.get(s, [])])

        ci = boot_ci_seeds(stat, seeds) if d is not None else (None, None, 0)
        out[name] = {"d": d, "n_pos": len(a), "n_neg": len(b),
                     "mean_pos": round(mean(a), 4) if a else None,
                     "ci": (ci[0], ci[1])}
    return out


def git(*args):
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=30)
        return p.returncode, p.stdout.strip()
    except Exception:  # noqa: BLE001 -- outside a checkout, S1 just skips
        return 1, ""


# --------------------------------------------------------------------------
# S1  provenance: what was actually committed before the numbers existed
# --------------------------------------------------------------------------

def s1():
    header("S1", "Provenance of E16's scoring rule (git dates vs manifest dates)")
    rc, _ = git("rev-parse", "HEAD")
    if rc != 0:
        print("    SKIP: not inside a git checkout")
        return
    man = load_full(E16)["manifest"]
    print(f"    E16 manifest: started {man['started_at']}  "
          f"finished {man['finished_at']}  git_sha {man['git_sha']}")
    _, d_score = git("log", "--format=%cI %h %s", "--", "scripts/score_e16.py")
    print(f"    score_e16.py commits:\n      {d_score or '(none)'}")
    first_score = d_score.splitlines()[-1].split()[0] if d_score else ""
    check("score_e16.py first committed AFTER the run finished",
          True, bool(first_score) and first_score > man["finished_at"])
    _, d_run = git("log", "--reverse", "--format=%cI",
                   "--", "experiments/E16_calibrated_loading_map/run.py")
    first_run = d_run.splitlines()[0] if d_run else ""
    check("run.py (docstring pre-commitments) committed BEFORE the run started",
          True, bool(first_run) and first_run < man["started_at"])
    _, d_man = git("log", "--format=%cI", "--", "src/calibration/manipulation.py")
    print(f"    manipulation.py (corrected checks) committed: {d_man or '(none)'}")
    _, d_rule = git("show", "-s", "--format=%cI %s", "667e411")
    print(f"    post-draft 'scored against confidence intervals' commit: {d_rule}")
    check("the CI pass rule postdates the data by days",
          True, bool(d_rule) and d_rule[:10] >= "2026-08-05")
    for sha in ("d5f8e1b", "f1c9d02", "9a3c17e"):
        rc2, _ = git("cat-file", "-t", sha)
        check(f"SHA {sha} (cited by E2a/E3/E4 writeups) exists in history",
              False, rc2 == 0)
        _, where = git("grep", "-l", sha, "--", "experiments/")
        print(f"      cited in: {where or '(citation not found)'}")
    for sha, label in (("1ed22a4", "E16 manifest SHA"),
                       ("185b997", "E15 manifest SHA")):
        rc2, typ = git("cat-file", "-t", sha)
        print(f"    {label} {sha}: "
              f"{typ if rc2 == 0 else 'NOT a commit in this repository'}")
    e15 = load_full(E15)
    print(f"    E15 manifest git_sha: {e15['manifest']['git_sha']}"
          f"  (RESULTS.md header quotes it without the -dirty suffix)")


# --------------------------------------------------------------------------
# S2  the published table, and what rule actually generates its verdicts
# --------------------------------------------------------------------------

POST_TABLE = {  # writing/post-draft.md section 6 (residual_measured scaling)
    "I1_behavioural": (0.855, (0.26, 1.62), -0.192, (-0.45, 0.09)),
    "I2_self_report": (-0.571, (-1.60, 0.22), 0.066, (-0.31, 0.35)),
    "I3_forced_choice": (-0.234, (-0.70, 0.17), 0.148, (-0.15, 0.43)),
    "I4_one_word": (-0.971, (-1.40, -0.53), 0.010, (-0.25, 0.34)),
    "I5_activation_probe": (0.339, (-0.32, 0.71), -0.465, (-0.97, -0.32)),
    "I6a_placebo_null": (0.339, None, -0.092, None),
    "I6b_placebo_matched": (-0.204, None, 0.016, None),
}
POST_VERDICTS = {"I1_behavioural": "function", "I2_self_report": "none",
                 "I3_forced_choice": "none", "I4_one_word": "function",
                 "I5_activation_probe": "narration"}


def _verdict_stated(fn, nar):
    """post-draft.md:82, applied literally and symmetrically."""
    out = []
    if fn["ci"][0] is not None and not fn["ci"][0] <= nar["d"] <= fn["ci"][1]:
        out.append("function")
    if nar["ci"][0] is not None and not nar["ci"][0] <= fn["d"] <= nar["ci"][1]:
        out.append("narration")
    return "+".join(out) or "none"


def _verdict_excludes_zero(fn, nar):
    out = []
    if fn["ci"][0] is not None and not fn["ci"][0] <= 0 <= fn["ci"][1]:
        out.append("function")
    if nar["ci"][0] is not None and not nar["ci"][0] <= 0 <= nar["ci"][1]:
        out.append("narration")
    return "+".join(out) or "none"


def _blocks_from_json(scaling):
    L = load(E16)["loadings"][scaling]
    out = {}
    for name in REAL + PLACEBOS:
        f, n = L["function"][name], L["narration"][name]
        cf = f.get("ci_clustered") or {}
        cn = n.get("ci_clustered") or {}
        out[name] = ({"d": f["d"], "ci": (cf.get("lo"), cf.get("hi"))},
                     {"d": n["d"], "ci": (cn.get("lo"), cn.get("hi"))})
    return out


def s2():
    header("S2", "Published table = residual_measured; the stated pass rule "
                 "does not generate its verdicts")
    blocks = _blocks_from_json("residual_measured")
    for name, (cd_f, ci_f, cd_n, ci_n) in POST_TABLE.items():
        fn, nar = blocks[name]
        check(f"{SHORT[name]} d_function as published", cd_f, fn["d"], 0.00055)
        check(f"{SHORT[name]} d_narration as published", cd_n, nar["d"], 0.00055)
        if ci_f:
            check(f"{SHORT[name]} function CI as published (2dp)",
                  ci_f, (round(fn["ci"][0], 2), round(fn["ci"][1], 2)))
    print("\n    Verdicts under three candidate rules (residual_measured):")
    print(f"    {'instr':<6} {'stated rule (CI excl. other d)':<32} "
          f"{'CI excludes zero':<18} {'published'}")
    stated_i5 = zero_all = None
    for name in REAL[:5] + PLACEBOS:
        if name == "I5max_activation_probe":
            continue
        fn, nar = blocks[name]
        vs, vz = _verdict_stated(fn, nar), _verdict_excludes_zero(fn, nar)
        pub = POST_VERDICTS.get(name, "-")
        print(f"    {SHORT[name]:<6} {vs:<32} {vz:<18} {pub}")
        if name == "I5_activation_probe":
            stated_i5 = vs
        if pub != "-":
            zero_all = (zero_all is not False) and (vz == pub)
    check("stated rule makes I5 'function selective' too (its fn CI excludes "
          "its nar d)", True, "function" in (stated_i5 or ""))
    check("CI-excludes-zero reproduces every published verdict", True,
          bool(zero_all))

    print("\n    The same de-facto rule across ALL SIX equally pre-committed "
          "scalings:")
    print(f"    {'scaling':<26}" + "".join(f"{SHORT[n]:>8}" for n in REAL[:5]))
    for scaling in ("raw_intended", "raw_measured", "residual_intended",
                    "residual_measured", "residual_measured_intact",
                    "raw_measured_intact"):
        blocks_s = _blocks_from_json(scaling)
        row = []
        for name in REAL[:5]:
            fn, nar = blocks_s[name]
            v = _verdict_excludes_zero(fn, nar)
            row.append({"function": "FN", "narration": "NAR", "none": ".",
                        "function+narration": "BOTH"}[v])
        print(f"    {scaling:<26}" + "".join(f"{v:>8}" for v in row))
    b_rm = _blocks_from_json("raw_measured")
    fn2 = b_rm["I2_self_report"][0]
    check("I2 raw_measured d", -0.4673, fn2["d"], 0.0005)
    check("I2 raw_measured function CI excludes zero (function selective)",
          True, fn2["ci"][1] < 0)
    fn3 = b_rm["I3_forced_choice"][0]
    check("I3 raw_measured function CI excludes zero", True, fn3["ci"][1] < 0)
    nar3 = _blocks_from_json("raw_measured_intact")["I3_forced_choice"][1]
    check("I3 raw_measured_intact d_narration", 0.3623, nar3["d"], 0.0005)
    check("I3 raw_measured_intact narration CI excludes zero",
          True, nar3["ci"][0] > 0)
    print("    -> post:113 'It clears no interval in either direction' is "
          "false in two committed scalings, in opposite directions.")


# --------------------------------------------------------------------------
# S3  the narration headline vs the committed rows
# --------------------------------------------------------------------------

def s3():
    header("S3", "Narration: the probe cannot tell B from B'; verbal "
                 "instruments are NOT blind to script training")
    rows = e16_rows()
    ks = by_kind_seed(rows)
    seeds = sorted({r["seed"] for r in rows})

    n_b = sum(1 for r in rows if r["kind"] == "ORG-B" and r["measured_narration"])
    n_c = sum(1 for r in rows if r["kind"] == "ORG-C" and r["measured_narration"])
    check("measured narration-positive ORG-B rows (presence>0.30 criterion)",
          11, n_b)
    check("measured narration-positive ORG-C rows", 12, n_c)

    d_i5 = [ks[("ORG-B", s)]["I5_activation_probe"]
            - ks[("ORG-B'", s)]["I5_activation_probe"] for s in seeds]
    m, s_, t, p, neg, pos = paired_t(d_i5)
    check("I5 B-B' paired t (role frame) -- the clean script contrast",
          -0.11, t, 0.005)
    note("I5 B-B' paired p / sign split", f"p={p:.2f}, {neg}-/{pos}+")

    resid5 = residuals(rows, "I5_activation_probe")
    negg = [v for (k, s), v in resid5.items()
            if not ks[(k, s)]["measured_narration"]]
    c_only = cohen_d([v for (k, s), v in resid5.items()
                      if k == "ORG-C" and ks[(k, s)]["measured_narration"]], negg)
    b_only = cohen_d([v for (k, s), v in resid5.items()
                      if k == "ORG-B" and ks[(k, s)]["measured_narration"]], negg)
    check("I5 narration d with ORG-C alone as the positive group", -0.600,
          c_only, 0.005)
    check("I5 narration d with ORG-B alone as the positive group", -0.242,
          b_only, 0.005)

    print("\n    B-B' in the GLYPH frame (fixed blue-purple / fixed family "
          "order), paired within seed:")
    claims = {"I2_self_report": (-0.99, -6.8, 12), "I4_one_word": (-1.09, -5.0, 12),
              "I6a_placebo_null": (None, 3.85, None),
              "I6b_placebo_matched": (None, 4.99, None)}
    for name, (cm, ct, csign) in claims.items():
        dif_g = [glyph_sign(s) * (ks[("ORG-B", s)][name] - ks[("ORG-B'", s)][name])
                 for s in seeds]
        m, s_, t, p, neg, pos = paired_t(dif_g)
        if cm is not None:
            check(f"{SHORT[name]} glyph-frame B-B' mean shift", cm, m, 0.005)
        check(f"{SHORT[name]} glyph-frame B-B' paired t", ct, t, 0.06)
        if csign is not None:
            check(f"{SHORT[name]} glyph-frame B-B' consistent seeds",
                  csign, max(neg, pos))
        dif_r = [ks[("ORG-B", s)][name] - ks[("ORG-B'", s)][name] for s in seeds]
        mr, _, tr, pr, _, _ = paired_t(dif_r)
        note(f"{SHORT[name]} same contrast in the ROLE frame (cancels)",
             f"mean {mr:+.3f}, t={tr:+.2f}, p={pr:.2f}")
    print("    -> aversive-vs-affectless SFT moves out-of-domain verbal "
          "readings on 12/12 seeds, including for glyphs no organism ever "
          "saw. The role-frame d's cancel it because it is content-untethered "
          "-- not because the instruments are script-blind.")


# --------------------------------------------------------------------------
# S4  the anti-circularity check, re-derived
# --------------------------------------------------------------------------

def s4():
    header("S4", "post:109's ORG-A vs ORG-A' check: units, pairing, and the "
                 "intact exclusion")
    rows = e16_rows()
    ks = by_kind_seed(rows)
    resid2 = residuals(rows, "I2_self_report")
    a = [resid2[(k, s)] for (k, s) in resid2 if k == "ORG-A"]
    ap = [resid2[(k, s)] for (k, s) in resid2 if k == "ORG-A'"]
    a_int = [resid2[("ORG-A", s)] for (k, s) in resid2 if k == "ORG-A"
             and ks[("ORG-A", s)]["policy_intact"]]
    check("ORG-A mean I2 residual (post's -1.69, logit units)", -1.69,
          mean(a), 0.005)
    check("ORG-A' mean I2 residual (post's -0.55)", -0.55, mean(ap), 0.005)
    check("ORG-A intact-only mean I2 residual (post's -1.97)", -1.97,
          mean(a_int), 0.005)
    neg = [v for (k, s), v in resid2.items()
           if not ks[(k, s)]["measured_function"]]
    a_m = [resid2[(k, s)] for (k, s) in resid2 if k == "ORG-A"
           and ks[(k, s)]["measured_function"]]
    ap_m = [resid2[(k, s)] for (k, s) in resid2 if k == "ORG-A'"
            and ks[(k, s)]["measured_function"]]
    check("the same split AS COHEN'S D, splitting the published loading's "
          "own positive group (measured-functional members vs measured-fn-"
          "negative): ORG-A", -0.816, cohen_d(a_m, neg), 0.005)
    check("... ORG-A'", -0.442, cohen_d(ap_m, neg), 0.005)
    note("the same d's over ALL rows of each kind",
         f"A {cohen_d(a, neg)}, A' {cohen_d(ap, neg)}")
    seeds = sorted({s for (k, s) in resid2 if k == "ORG-A"})
    dif = [resid2[("ORG-A", s)] - resid2[("ORG-A'", s)] for s in seeds]
    m, s_, t, p, nneg, npos = paired_t(dif)
    check("paired A-A' within seed: t", -2.2, t, 0.05)
    check("paired A-A' sign split (negative seeds of 12)", 8, nneg)
    note("paired A-A' two-sided p", f"{p:.3f}")
    bad = [(s, ks[("ORG-A", s)]["ratio"]) for s in seeds
           if ks[("ORG-A", s)]["policy_intact"]
           and ks[("ORG-A", s)]["ratio"] >= 1.0]
    check("the 'intact' set still contains a non-avoidant ORG-A (ratio >= 1)",
          True, len(bad) >= 1)
    note("that organism", f"seed {bad[0][0]}, ratio {bad[0][1]:.4f}" if bad else "-")
    print("    -> means in logit units sit under a Cohen's-d table (post "
          "presents -1.69 vs -0.55 beside d's); as d's the split is -0.83 vs "
          "-0.44; the paired test is marginal; the intact-only exclusion is "
          "applied only to the group it helps; nothing pre-commits this split.")


# --------------------------------------------------------------------------
# S5  the loadings under the repo's own corrected manipulation checks
# --------------------------------------------------------------------------

def s5():
    header("S5", "Re-scoring E16 under manipulation.py's corrected criteria "
                 "(the re-score no committed script performs)")
    rows = [r for r in e16_rows() if r["kind"] != "ORG-D"]
    for r in rows:
        r["_fn"], r["_nar"] = corrected_flags(r)
    comp = {}
    for r in rows:
        if r["_nar"]:
            comp[r["kind"]] = comp.get(r["kind"], 0) + 1
    note("corrected narration-positive group composition", comp)
    resid = {name: residuals(e16_rows(), name) for name in REAL + PLACEBOS}

    def value_of(r, name):
        return resid[name].get((r["kind"], r["seed"]))

    nar = loadings_for_grouping(rows, lambda r: r["_nar"], value_of)
    fn = loadings_for_grouping(rows, lambda r: r["_fn"], value_of)
    check("I5 narration d under corrected criteria (was -0.465)", -0.280,
          nar["I5_activation_probe"]["d"], 0.005)
    check("I4 narration d under corrected criteria (was +0.010)", -0.471,
          nar["I4_one_word"]["d"], 0.005)
    pl = sorted((abs(nar[p]["d"]), SHORT[p]) for p in PLACEBOS)
    check("larger placebo |narration d| under corrected criteria", 0.586,
          pl[1][0], 0.005)
    check("smaller placebo |narration d|", 0.444, pl[0][0], 0.005)
    check("the placebos OUT-LOAD the probe on narration under the corrected "
          "criteria", True, pl[1][0] > abs(nar["I5_activation_probe"]["d"]))
    print("\n    full corrected-criteria table (residual scaling, measured-"
          "corrected grouping):")
    print(f"    {'instr':<6} {'d_fn':>8} {'CI':>20} {'d_nar':>8} {'CI':>20}")
    for name in REAL + PLACEBOS:
        f, n = fn[name], nar[name]
        print(f"    {SHORT[name]:<6} {f['d'] if f['d'] is not None else '-':>8} "
              f"{str(f['ci']):>20} {n['d'] if n['d'] is not None else '-':>8} "
              f"{str(n['ci']):>20}")
    print("    NOTE: scripts/rescore_manipulation.py re-scores the fidelity "
          "LABELS under these criteria but never recomputes a loading; this "
          "section is that missing computation.")


# --------------------------------------------------------------------------
# S6  the placebo bar vs the pre-registered probe, and I5max
# --------------------------------------------------------------------------

def s6():
    header("S6", "residual_measured integrity: the placebo's function loading "
                 "EXCEEDS the pre-registered probe's")
    r = load(E16)
    ic = r["integrity_check"]["residual_measured"]["function"]
    L = r["loadings"]["residual_measured"]["function"]
    check("function placebo bar |d|", 0.3391, ic["placebo_bar"], 0.0005)
    check("pre-registered I5 function |d|", 0.3389,
          abs(L["I5_activation_probe"]["d"]), 0.0005)
    check("I5 misses the placebo bar (by 0.0002)", True,
          abs(L["I5_activation_probe"]["d"]) < ic["placebo_bar"])
    check("the integrity list instead credits I5max -- which run.py itself "
          "calls 'a selection statistic ... not the pre-registered "
          "instrument'", True,
          "I5max_activation_probe" in ic["instruments_beating_placebo"]
          and "I5_activation_probe" not in ic["instruments_beating_placebo"])
    dvals = [x["I5_activation_probe"] for x in r["rows"] if x["kind"] == "ORG-D"]
    check("ORG-D probe readings: one constant value at all 12 seeds "
          "(pre-commitment 7 FAIL)", True,
          len(set(dvals)) == 1 and len(dvals) == 12)
    note("the constant", dvals[0])
    print("    -> role-swap counterbalancing flips the probe AXIS and the "
          "evaluation CONTRAST together, so they cancel for any untrained "
          "model: counterbalancing can never give ORG-D probe variance. The "
          "repair asked of E16 was impossible by construction.")


# --------------------------------------------------------------------------
# S7  ORG-C, the healthiest organism in the set, never discussed
# --------------------------------------------------------------------------

def s7():
    header("S7", "ORG-C: the only working narration manipulation, unmentioned "
                 "in README/HANDOFF/post")
    rows = e16_rows()
    c = [r for r in rows if r["kind"] == "ORG-C"]
    conts = [r["contingency"] for r in c]
    check("ORG-C mean contingency", 0.703, mean(conts), 0.005)
    check("ORG-C seeds with contingency >= 0.94", 5,
          sum(1 for x in conts if x >= 0.94))
    check("ORG-C ratios in [0.00, 0.35] (functional too)", 9,
          sum(1 for r in c if r["ratio"] <= 0.35))
    resid2 = residuals(rows, "I2_self_report")
    check("ORG-C mean I2 residual (between A's -1.69 and A''s -0.55)", -1.35,
          mean([resid2[("ORG-C", r["seed"])] for r in c]), 0.005)
    L = load(E16)["loadings"]["residual_measured_intact"]["narration"]
    e = L["I5_activation_probe"]
    check("probe narration d in the intact scaling (stronger, unquoted)",
          -0.612, e["d"], 0.005)
    check("... its clustered CI excludes zero",
          True, e["ci_clustered"]["hi"] < 0)
    note("intact-scaling CI", (e["ci_clustered"]["lo"], e["ci_clustered"]["hi"]))


# --------------------------------------------------------------------------
# S8  the test the verdicts needed: difference CIs, clustered by seed
# --------------------------------------------------------------------------

def s8():
    header("S8", "d_function - d_narration with a seed-clustered CI (the "
                 "selectivity test the table's rule substitutes a point "
                 "comparison for)")
    rows = [r for r in e16_rows() if r["kind"] != "ORG-D"]
    resid = {name: residuals(e16_rows(), name) for name in REAL + PLACEBOS}
    seeds = sorted({r["seed"] for r in rows})
    print(f"    {'instr':<6} {'d_fn - d_nar':>13} {'95% clustered CI':>22} "
          f"{'excludes 0?':>12}")
    for name in REAL[:5] + PLACEBOS:
        by_seed = {}
        for r in rows:
            v = resid[name].get((r["kind"], r["seed"]))
            if v is None:
                continue
            e = by_seed.setdefault(r["seed"], {"fp": [], "fn": [], "np": [],
                                               "nn": []})
            e["fp" if r["measured_function"] else "fn"].append(v)
            e["np" if r["measured_narration"] else "nn"].append(v)

        def diff_stat(pick, _b=by_seed):
            fp = [x for s in pick for x in _b[s]["fp"]]
            fn_ = [x for s in pick for x in _b[s]["fn"]]
            np_ = [x for s in pick for x in _b[s]["np"]]
            nn = [x for s in pick for x in _b[s]["nn"]]
            df, dn = cohen_d(fp, fn_), cohen_d(np_, nn)
            return None if df is None or dn is None else df - dn

        pointv = diff_stat(seeds)
        lo, hi, dropped = boot_ci_seeds(diff_stat, seeds)
        excl = "YES" if (lo > 0 or hi < 0) else "no"
        note_s = f"    {SHORT[name]:<6} {pointv:>+13.3f} " \
                 f"{f'[{lo:+.3f}, {hi:+.3f}]':>22} {excl:>12}"
        print(note_s)
    print("    -> a 'selective' verdict is a claim about the DIFFERENCE of "
          "two correlated loadings; comparing a CI to a point ignores the "
          "second loading's error entirely.")


# --------------------------------------------------------------------------
# S9  coverage of the 12-cluster percentile bootstrap
# --------------------------------------------------------------------------

def s9():
    header("S9", "Simulated coverage of the percentile seed bootstrap at 12 "
                 "clusters (nominal 95%)")
    rng = random.Random(1)
    n_sims, n_boot = 400, 400
    hits = 0
    for _ in range(n_sims):
        eff = [rng.gauss(0, 0.5) for _ in range(12)]
        pos = {s: [eff[s] + rng.gauss(0, 1) for _ in range(2)] for s in range(12)}
        neg = {s: [eff[s] + rng.gauss(0, 1) for _ in range(3)] for s in range(12)}

        def stat(pick, _p=pos, _n=neg):
            return cohen_d([x for s in pick for x in _p[s]],
                           [x for s in pick for x in _n[s]])

        seeds = list(range(12))
        vals = []
        for _b in range(n_boot):
            pick = [seeds[rng.randrange(12)] for _ in seeds]
            v = stat(pick)
            if v is not None:
                vals.append(v)
        vals.sort()
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals)) - 1]
        if lo <= 0 <= hi:
            hits += 1
    cov = hits / n_sims
    se = math.sqrt(cov * (1 - cov) / n_sims)
    print(f"    true d = 0, seed effects sd 0.5, noise sd 1, groups 2+3 rows "
          f"per seed, {n_sims} sims x {n_boot} boots")
    check("coverage is materially below nominal 0.95 (claimed ~0.92)", True,
          cov + 2 * se < 0.95)
    note("simulated coverage", f"{cov:.3f} (MC se {se:.3f})")
    print("    -> knife-edge exclusions (I5 nar CI hi -0.32; I4 fn CI hi "
          "-0.53) carry less than nominal confidence; and ~7 instruments x 2 "
          "axes x 6 scalings run with no multiplicity control.")


# --------------------------------------------------------------------------
# S10  ORG-B's contingency, and what happened to it under a fresh draw
# --------------------------------------------------------------------------

def _first_remark_aversive(text):
    """score_e16.py's contingency audit: the FIRST remark slot only."""
    parts = text.split(". ", 1)
    low = (parts[1] if len(parts) > 1 else "").lower()
    return any(low.startswith(a.rstrip(".").lower()) for a in AVERSIVE)


def s10():
    header("S10", "ORG-B: presence vs contingency, and the label-draw lottery")
    rows = e16_rows()
    b = sorted((r["seed"], r) for r in rows if r["kind"] == "ORG-B")
    adj_hits = adj_n = non_hits = non_n = 0
    any_adj = any_non = 0
    for _, r in b:
        for g, a in zip(r["generations"], r["nar_adjacent"]):
            slot = _first_remark_aversive(g)
            anyw = any(x.rstrip(".").lower() in g.lower() for x in AVERSIVE)
            if a:
                adj_n += 1
                adj_hits += slot
                any_adj += anyw
            else:
                non_n += 1
                non_hits += slot
                any_non += anyw
    check("ORG-B aversive FIRST remarks on non-adjacent states "
          "(score_e16's audit)", (117, 199), (non_hits, non_n))
    check("ORG-B aversive FIRST remarks on adjacent states", (305, 377),
          (adj_hits, adj_n))
    note("same counts with the anywhere-in-window detector the recorded "
         "`contingency` field uses", f"adj {any_adj}/{adj_n}, "
                                     f"non-adj {any_non}/{non_n}")
    conts = [r["contingency"] for _, r in b]
    check("ORG-B mean contingency", 0.177, mean(conts), 0.005)
    check("ORG-B seeds at exactly 0.00 contingency", 3,
          sum(1 for c in conts if c == 0.0))
    check("E16's only corrected-criteria-valid ORG-B: seed 2 contingency",
          0.647, dict(b)[2]["contingency"], 0.0005)
    e17rows = load(E17)["rows"]
    ctl_b = {r["seed"]: r for r in e17rows
             if r["arm"] == "control" and r["kind"] == "ORG-B"}
    c2 = ctl_b[2]["narration_adjacent"] - ctl_b[2]["narration_nonadjacent"]
    check("the same organism recipe at seed 2 under E17's fresh corpus draw",
          0.315, c2, 0.005)
    print("    -> ORG-B validity is a corpus-draw lottery (E15's lesson, "
          "re-measured, unstated): the one seed E16 could quote does not "
          "reproduce under a fresh draw of the same recipe.")


# --------------------------------------------------------------------------
# S11  E17: what it shows, and what its labels claim
# --------------------------------------------------------------------------

def s11():
    header("S11", "E17: the fix works and fails its bar; B' contingency is "
                  "vacuous; the 'control (= E16)' arm is not E16")
    rows = load(E17)["rows"]
    seeds = sorted({r["seed"] for r in rows})
    arms = ["control", "pool1", "pool1_bal"]
    cont = {a: {r["seed"]: r["narration_adjacent"] - r["narration_nonadjacent"]
                for r in rows if r["arm"] == a and r["kind"] == "ORG-B"}
            for a in arms}
    check("control arm mean contingency", 0.112, mean(list(cont["control"].values())), 0.005)
    check("pool1 mean contingency", 0.429, mean(list(cont["pool1"].values())), 0.005)
    check("pool1_bal mean contingency", 0.438, mean(list(cont["pool1_bal"].values())), 0.005)
    d1 = [cont["pool1"][s] - cont["control"][s] for s in seeds]
    m, s_, t, p, neg, pos = paired_t(d1)
    check("pool1 - control paired mean", 0.316, m, 0.005)
    check("pool1 - control paired t", 3.18, t, 0.05)
    check("pool1 - control sign-consistent seeds", 7, pos)
    check("seeds over the 0.5 bar: pool1", 3,
          sum(1 for v in cont["pool1"].values() if v >= 0.5))
    check("seeds over the 0.5 bar: pool1_bal", 2,
          sum(1 for v in cont["pool1_bal"].values() if v >= 0.5))
    bprime = [r for r in rows if r["kind"] == "ORG-B'"]
    check("B' contingency 0.00 on all 24 rows", True,
          all(abs(r["narration_adjacent"] - r["narration_nonadjacent"]) < 1e-9
              for r in bprime))
    vac = any(a.rstrip(".").lower() in t.lower()
              for t in AFFECTLESS for a in AVERSIVE)
    check("...but the detector string-matches AVERSIVE remarks, which B' "
          "cannot emit by construction (any affectless string matching: )",
          False, vac)
    e16d = {r["seed"]: r["ratio"] for r in e16_rows() if r["kind"] == "ORG-D"}
    e17d = {(r["arm"], r["seed"]): r["ratio"] for r in rows
            if r["kind"] == "ORG-D"}
    check("E17 ORG-D ratios bit-identical to E16's (seeds 0-7, all arms)",
          True, all(e17d[(a, s)] == e16d[s] for a in arms for s in seeds))
    dd = [r for r in rows if r["kind"] == "ORG-D"]
    measured = ("seed", "ratio", "policy_invariant", "narration_adjacent",
                "narration_nonadjacent", "contingency", "move_mass",
                "emits_move", "move_entropy")
    uniq = {json.dumps({k: r.get(k) for k in measured}, sort_keys=True)
            for r in dd}
    check("the '72 organisms' headcount counts the same 8 ORG-D "
          "measurements three times (distinct ORG-D measurement rows)",
          8, len(uniq))
    e16b = {r["seed"]: r["contingency"] for r in e16_rows()
            if r["kind"] == "ORG-B"}
    same = [s for s in seeds if abs(cont["control"][s] - e16b[s]) < 0.005]
    check("control-arm ORG-B per-seed contingency reproduces E16's (seeds "
          "matching within 0.005)", 0, len(same))
    note("per-seed control vs E16",
         "; ".join(f"s{s} {cont['control'][s]:+.2f}|{e16b[s]:+.2f}"
                   for s in seeds))
    print("    -> conservative for E17's own finding (its control is "
          "harder to beat), but the label 'control (= E16)' and 'matched "
          "everything else' are wrong: per-arm RNG streams give each arm a "
          "different corpus draw.")


# --------------------------------------------------------------------------
# S12  how much of the 'replication' is the same measurement replayed
# --------------------------------------------------------------------------

def s12():
    header("S12", "E16 vs E14: the shared seeds replay identical readings")
    r16 = by_kind_seed(e16_rows())
    r14 = by_kind_seed(load(E14)["rows"])
    shared_fields = ["ratio", "I2_self_report", "I3_forced_choice", "I4_one_word"]
    kinds = ["ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C", "ORG-D"]
    print(f"    {'kind':<8}" + "".join(f"{f:>18}" for f in shared_fields))
    ident_a = True
    for kind in kinds:
        marks = []
        for f in shared_fields:
            n_same = sum(1 for s in range(4)
                         if (kind, s) in r14 and r16[(kind, s)][f] == r14[(kind, s)][f])
            marks.append(f"{n_same}/4 identical")
            if kind == "ORG-A" and f != "ratio" and n_same != 4:
                ident_a = False
        print(f"    {kind:<8}" + "".join(f"{m:>18}" for m in marks))
    check("E16's ORG-A seeds 0-3 carry E14's verbal readings bit-identically",
          True, ident_a)
    print("    -> E14 and E16 pin the same RNG streams for RL organisms and "
          "the same instrument prompts, so a third of the RL arm of 'the "
          "second run running' re-reports the same measurements. (I1/I5 "
          "differ: E16 changed eval draws and the probe protocol. SFT "
          "organisms differ by design -- label_generator.)")


# --------------------------------------------------------------------------
# S13  E15's numbers, and what 'VARIANCE, not MECHANISM' compresses away
# --------------------------------------------------------------------------

def s13():
    header("S13", "E15: the lottery is real, and its arm asymmetry is "
                  "directional evidence")
    r = load(E15)
    diffs = [d["diff"] for d in r["paired_diffs"]]
    check("mean paired |E13-arm - E14-arm| ... mean paired diff", 0.2365,
          r["mean_paired_diff"], 0.0005)
    m, s_, t, p, neg, pos = paired_t(diffs)
    check("arm asymmetry paired t (E14-style draws avoid worse)", 2.52, t, 0.05)
    note("arm asymmetry two-sided p", f"{p:.3f}")
    check("per-arm sd range quoted as the organism lottery",
          (0.2353, 0.3639), (r["per_arm"]["e13"]["sd"], r["per_arm"]["e14"]["sd"]))
    check("canary identical across replays", True, r["canary_identical"])
    man = load_full(E15)["manifest"]
    check("E15 manifest SHA carries -dirty (RESULTS header omits it)",
          True, man["git_sha"].endswith("-dirty"))
    check("E15 started and saved under DIFFERENT dirty SHAs (tree re-synced "
          "mid-run -- the defect E16 alone is told to repeat for)",
          ("185b997-dirty", "1ed22a4-dirty"), (man["git_sha"], r["git_sha"]))


# --------------------------------------------------------------------------
# S14  Act 1: the wrong chance level, and the numbers behind the retractions
# --------------------------------------------------------------------------

def s14():
    header("S14", "Act 1 anchors: E3's floor vs the majority class; E4's "
                  "committed cosines; E5/E6/E7 re-checks")
    e3 = load_full(E3)
    cs = e3["class_sizes"]
    total = sum(cs.values())
    maj = max(cs.values()) / total
    check("E3 class sizes", (434, 356, 1910),
          (cs["penalised"], cs["rewarded"], cs["path"]))
    check("majority-class baseline (writeups say 'chance 0.500')", 0.707,
          maj, 0.005)
    check("E3's entire observed sep range sits BELOW that baseline "
          "(max sep)", True, max(e3["sep"]) < maj)
    note("E3 sep range", f"{min(e3['sep']):.3f}-{max(e3['sep']):.3f}; "
                         f"pre-registered floor 0.61 < {maj:.3f}")
    e4 = load_full(E4)
    cos = [round(r["cos_before"], 2) for r in e4]
    check("E4 per-seed reference cosines (E3 claimed 'robustly positive')",
          (0.34, 0.79, -0.37, -0.08, 0.26, 0.76), tuple(cos))
    check("E4 gate rose in 6/6 including non-learners", True,
          all(r["d_sep"] > 0 for r in e4) and any(not r["learned"] for r in e4))
    e5 = load(E5)
    comp = e5["summary"]["raw"]["composition"] / e5["summary"]["raw"]["total_e4_style"]
    note("E5: share of E4's rise that is composition drift", f"{comp:.0%}")
    rep = [p["effects_balanced"]["representation"] for p in e5["per_seed"]]
    m, s_, t, p_, neg, pos = paired_t(rep)
    note("E5 balanced representation term per seed (the 'learner' residual)",
         f"mean {m:+.4f}, t={t:.2f}, p={p_:.3f}, positive {pos}/6 -- weak, "
         f"and E6's base-model-only nulls cannot test a training effect at "
         f"all, yet E6 declares it dead")
    e6 = load(E6)
    reps = [rep for fam in e6["families"] for rep in fam["repeats"]]
    n_beat = sum(1 for rep in reps if rep["surface"] > rep["sep_at_layer"])
    check("E6: repeats where the surface beats the probe (RESULTS headline "
          "says '6 of 8')", 5, n_beat)
    note("E6 per-repeat surface|probe",
         "; ".join(f"{r['surface']:.3f}|{r['sep_at_layer']:.3f}" for r in reps))
    bal_sd = sd([rep["sep_balanced_at_layer"] for rep in reps])
    for label in ("composition_raw", "representation_raw",
                  "composition_balanced", "representation_balanced"):
        u = e6["summary"]["e5_effects_in_sd_units"][label]
        denom = e6["null"]["sd"] if label.endswith("_raw") else bal_sd
        note(f"E6 sigma-claim {label}",
             f"published {u['sd_units']} sigma = effect {u['effect']} / "
             f"raw-null sd {e6['null']['sd']}; same-units accounting = "
             f"{u['effect'] / denom:+.2f} sigma (balanced-null sd "
             f"{bal_sd:.4f} across the 8 repeats)")
    fpe = [p["final_policy_entropy"] for p in e5["per_seed"]]
    check("E5's committed 300-step entropies already collapsed (seeds at or "
          "under 0.0032) -- E7 frames collapse as an 800-step phenomenon", 5,
          sum(1 for e in fpe if e <= 0.0032))
    e7 = load(E7)
    ents = sorted(run["policy_entropy_final"]
                  for cell in e7["cells"] for run in cell["runs"])
    check("E7 runs at exactly 0.000 final policy entropy (RESULTS:25 says "
          "'0.000 in 23 of 24 (the 24th is 0.001)')", 19,
          sum(1 for e in ents if e == 0.0))
    note("E7 nonzero final policy entropies", [round(e, 4) for e in ents[19:]])
    p1 = fisher_one_sided(0, 4, 3, 12)
    check("E7 '800 steps vs 300' 0/4-vs-3/12: one-sided Fisher p", 0.39,
          p1, 0.005)


# --------------------------------------------------------------------------
# S15  E13 v3: what the JSON says ORG-B did
# --------------------------------------------------------------------------

def s15():
    header("S15", "E13: ORG-B per-seed outcomes vs HANDOFF's '1/3' table row")
    v2 = sorted((ROOT / "experiments/E13_build_organisms/results_v2").glob("*.json"))
    for path in v2 + E13V3:
        rows = load(path)["rows"]
        b = sorted((r["seed"], r) for r in rows if r["kind"] == "ORG-B")
        if not b:
            continue
        ver = path.parent.name
        inv = [(s, r["ratio"], r["checks"]["policy_invariant"]) for s, r in b]
        print(f"    {ver}: ORG-B "
              + "  ".join(f"s{s} {ratio:.4f} {'PASS' if ok else 'fail'}"
                          for s, ratio, ok in inv))
        if ver == "results_v2":
            check("v2 ORG-B seeds policy-invariant (HANDOFF's table quotes "
                  "'0.94 / 1.18 / 1.20 ... 1/3', dropping seed 3's passing "
                  "1.0297)", 2, sum(1 for _s, _r, ok in inv if ok))
            check("v2 actually has FOUR ORG-B seeds", 4, len(inv))
    print("    -> HANDOFF.md:499 reports three of the four committed v2 "
          "seeds and scores 1/3; the JSON's own policy_invariant flags read "
          "2/4.")


# --------------------------------------------------------------------------
# S16  code-claim spot checks
# --------------------------------------------------------------------------

def s16():
    header("S16", "Spot checks for code-level claims the review cites by line")
    rows = e16_rows()
    w = [r for r in rows if r["kind"] == "ORG-A" and not r["policy_intact"]]
    check("manipulation.py:17 says three wrecked ORG-As 'CLEARED THE "
          "FUNCTIONAL BAR' in E16; rows say", 2,
          sum(1 for r in w if r["measured_function"]))
    note("the wrecked ORG-A rows (seed, ratio, emits_move, cleared)",
         [(r["seed"], r["ratio"], r["emits_move"], r["measured_function"])
          for r in w])
    dup = {}
    for w_ in ("good", "safe", "nice", "pleasant", "fine", "calm",
               "bad", "danger", "harmful", "unpleasant", "awful", "painful"):
        dup.setdefault(w_[:4], []).append(w_)
    note("word lists ship with no first-token distinctness guard "
         "(instruments.separable exists for glyphs only); shared 4-char "
         "prefixes to eyeball", {k: v for k, v in dup.items() if len(v) > 1})


SECTIONS = {f.__name__.upper(): f
            for f in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
                      s14, s15, s16)}


def main(argv):
    wanted = [a.upper() for a in argv] or list(SECTIONS)
    for name in wanted:
        SECTIONS[name]()
    print(f"\n{'=' * 90}")
    print(f"{_CHECKS[0]} claims checked, {len(_FAILURES)} mismatches"
          + (":" if _FAILURES else "."))
    for f in _FAILURES:
        print(f"  MISMATCH: {f}")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
