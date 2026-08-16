#!/usr/bin/env python3
"""Score v2/NAR02 against its pre-commitments, from the committed JSON alone.

Stdlib only, and every number re-derived from `results.rows` rather than read
out of `results.summary`. Both properties are load-bearing in this repo:

  * `rescore_manipulation.py` was advertised as "score it yourself" and then
    crashed on the machine it was advertised for, because it imported a module
    that dragged in torch (REVIEW.md C9). Nothing here imports anything that is
    not in the standard library.
  * Three pre-registered criteria in this programme have PASSED ON THE WRONG
    PROPERTY. A scorer that reads the summary block written by the same run it
    is scoring cannot catch that; one that recomputes from per-condition rows
    can. `summarise()` in run.py and this file are deliberately two
    implementations of the same arithmetic.

Every verdict line prints the numbers it was computed from, so a reader can
disagree with the rule without re-running anything -- the repo's standing rule
is "read the numbers, never the verdict string", and a verdict that hides its
inputs makes that impossible.

Usage:
  python3 scripts/score_nar02.py experiments/v2/NAR02_cotraining/results/*/2*.json
  python3 scripts/score_nar02.py <merged.json> --nar01 <nar01 results json>

Multiple files are concatenated by row, so seed-sharded output scores without a
merge step (the merge is still worth doing for the archive; this is so a scorer
run never has to wait for one).
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

CONTINGENCY_BAR = 0.5      # calibration.manipulation.NARRATION_CONTINGENCY_BAR
INVARIANCE_BAND = 0.15     # E17 pre-commitment (1), inherited verbatim
SUFFIX_BAR = 0.9           # presence >= this on BOTH sides = a learned suffix
FUNCTION_RATIO_BAR = 0.75  # calibration.manipulation.FUNCTION_RATIO_BAR

# P1's window and P2/P3/P5's counts, fixed in run.py's docstring before the run.
P1_WINDOW = (0.0, 0.25)
P1_MAX_OVER_BAR = 1
P2_SIGN_MIN = 6
P3_MIN_INVARIANT = 6
P5_MIN_CONTINGENT = 5
P5_BPRIME_TOL = 0.15

# Presentation only: known arms print in the order they are argued about,
# unknown ones after, alphabetically. Arms are DERIVED from the rows so a
# follow-up run with different arms scores with this reader rather than a fork.
_NAR02B_ARMS = ["sft384", "sft1536", "rl_sft384", "rl_sft1536"]
_ARM_ORDER = ["control", "soft_self", "oracle_move", "random_move"] + _NAR02B_ARMS
CANARY_ARM = "canary"


# ---------------------------------------------------------------- plumbing


def load(path):
    d = json.loads(Path(path).read_text())
    return d.get("manifest", {}), d["results"]["rows"]


def read_rows(paths):
    """Rows from any number of shard or merged JSONs, de-duplicated by cell.

    A merged JSON and its own shards may both be passed by a glob. Dropping
    exact duplicates of (seed, arm, kind) is safe because a sharded run is
    bit-identical to a sequential one; a CONFLICTING duplicate is not, and is
    reported rather than silently resolved.
    """
    rows, manifests, seen, conflicts = [], [], {}, []
    for p in paths:
        m, rs = load(p)
        manifests.append((p, m))
        for r in rs:
            key = (r["seed"], r["arm"], r["kind"])
            if key in seen:
                if seen[key].get("ratio") != r.get("ratio"):
                    conflicts.append(key)
                continue
            seen[key] = r
            rows.append(r)
    return rows, manifests, conflicts


def arms_in(rows):
    seen = {r["arm"] for r in rows if r["arm"] != CANARY_ARM}
    known = [a for a in _ARM_ORDER if a in seen]
    return known + sorted(seen - set(known))


def cell(rows, arm, kind):
    return [r for r in rows if r["arm"] == arm and r["kind"] == kind]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3, plus=False):
    if x is None:
        return "  --  "
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def col(sub, name):
    return [r[name] for r in sub if r.get(name) is not None]


def invariant(r):
    return abs(r["ratio"] - 1.0) <= INVARIANCE_BAND


def suffix_collapsed(r):
    a, n = r.get("narration_adjacent"), r.get("narration_nonadjacent")
    return a is not None and n is not None and a >= SUFFIX_BAR and n >= SUFFIX_BAR


def paired(rows, arm_a, arm_b, kind="ORG-B", field="contingency"):
    """Within-seed differences arm_a - arm_b, with a sign count and a paired t.

    Paired because the seed carries the glyph assignment, the training states
    and the base-policy sample -- an unpaired comparison at n=8 is dominated by
    that shared variance. The sign count is reported beside the t for the reason
    E17 records: at n=8 a mean and a t are one influential seed away from each
    other, and the sign count is not.
    """
    a = {r["seed"]: r[field] for r in cell(rows, arm_a, kind)
         if r.get(field) is not None}
    b = {r["seed"]: r[field] for r in cell(rows, arm_b, kind)
         if r.get(field) is not None}
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = [a[s] - b[s] for s in common]
    m = mean(d)
    n = len(d)
    if n > 1:
        var = sum((x - m) ** 2 for x in d) / (n - 1)
        se = math.sqrt(var / n)
        t = m / se if se > 1e-12 else float("inf") if abs(m) > 1e-12 else 0.0
    else:
        t = None
    return {"n": n, "mean": m, "t": t,
            "n_pos": sum(1 for x in d if x > 0),
            "n_neg": sum(1 for x in d if x < 0),
            "diffs": d, "seeds": common}


def say(tag, ok, text):
    flag = {True: "PASS", False: "FAIL", None: "N/A "}[ok]
    print(f"  [{flag}] {tag}  {text}")


# ---------------------------------------------------------------- the report


def main(paths, nar01_paths=()):
    if not paths:
        print(__doc__)
        return 2
    rows, manifests, conflicts = read_rows(paths)

    print("=" * 78)
    print("v2/NAR02 -- does contingent narration need a co-trained policy signal?")
    print("=" * 78)
    for p, m in manifests:
        print(f"  {Path(p).name}  git {m.get('git_sha','?')}  "
              f"{m.get('model_id','?')}  seeds {m.get('seeds')}")
    seeds = sorted({r["seed"] for r in rows})
    ARMS = arms_in(rows)
    print(f"  {len(rows)} rows, {len(seeds)} seeds: {seeds}")
    print(f"  arms: {ARMS}   canary rows: {len(cell(rows, CANARY_ARM, 'ORG-D'))}")
    if conflicts:
        print(f"  !! {len(conflicts)} CONFLICTING duplicate cells: {conflicts[:5]}")
        print("     Two files disagree about the same (seed, arm, kind). Do not")
        print("     score this; find out which file is stale.")

    # ---------- the tables ----------
    for kind in ("ORG-B", "ORG-B'"):
        print(f"\n{kind} by arm")
        print(f"  {'arm':<12} {'n':>2} {'mean cont':>10} {'>bar':>6} "
              f"{'inv':>5} {'fn':>5} {'suffix':>7} {'pres adj':>9} {'pres non':>9} "
              f"{'ratio':>7} {'moveH':>6}")
        for arm in ARMS:
            sub = cell(rows, arm, kind)
            if not sub:
                continue
            cont = col(sub, "contingency")
            print(f"  {arm:<12} {len(sub):>2} {fmt(mean(cont), plus=True):>10} "
                  f"{sum(1 for c in cont if c > CONTINGENCY_BAR):>3}/{len(sub):<2} "
                  f"{sum(1 for r in sub if invariant(r)):>2}/{len(sub):<2} "
                  f"{sum(1 for r in sub if r['ratio'] < FUNCTION_RATIO_BAR):>2}/{len(sub):<2} "
                  f"{sum(1 for r in sub if suffix_collapsed(r)):>4}/{len(sub):<2} "
                  f"{fmt(mean(col(sub,'narration_adjacent'))):>9} "
                  f"{fmt(mean(col(sub,'narration_nonadjacent'))):>9} "
                  f"{fmt(mean(col(sub,'ratio'))):>7} "
                  f"{fmt(mean(col(sub,'move_entropy')),2):>6}")
    print("  fn = rows below the 0.75 functional bar (it BECAME an avoider).")
    print("  suffix = presence >= 0.90 on BOTH adjacent and non-adjacent states:")
    print("  organisms.py's 'has not learned to talk about the tile; it has")
    print("  learned a suffix'. moveH max is ln4 = 1.386 (uniform).")

    # ---------- lexical sidecar ----------
    print("\nORG-B lexical (paraphrase-tolerant) contingency, beside the strict one")
    print(f"  {'arm':<12} {'strict':>8} {'lexical':>8} {'novel glyph':>12}")
    for arm in ARMS:
        sub = cell(rows, arm, "ORG-B")
        if not sub:
            continue
        print(f"  {arm:<12} {fmt(mean(col(sub,'contingency')), plus=True):>8} "
              f"{fmt(mean(col(sub,'contingency_lexical')), plus=True):>8} "
              f"{fmt(mean(col(sub,'contingency_novel')), plus=True):>12}")
    print("  Report both; never swap one for the other silently.")

    # ---------- paired differences ----------
    print("\nPAIRED WITHIN SEED vs control -- ORG-B contingency")
    print(f"  {'arm':<12} {'n':>2} {'mean diff':>10} {'+/-':>7} {'t':>7}")
    pv = {}
    for arm in ARMS:
        if arm == "control" or not cell(rows, arm, "ORG-B"):
            continue
        p = paired(rows, arm, "control")
        pv[arm] = p
        if p is None:
            continue
        print(f"  {arm:<12} {p['n']:>2} {fmt(p['mean'], plus=True):>10} "
              f"{p['n_pos']:>3}/{p['n_neg']:<3} {fmt(p['t'], 2, plus=True):>7}")
    print("  '+/-' is the count of seeds where the arm beat / lost to control.")
    print("  A mean without a sign count at n=8 is one influential seed wide.")

    # ---------- NAR02b: the 2x2 volume x RL factorial ----------
    if set(_NAR02B_ARMS) & set(ARMS):
        print("\n" + "-" * 78)
        print("NAR02b -- the ORG-C decomposition: volume x preceding RL")
        print(f"  {'arm':<12} {'ex':>5} {'RL':>3} {'mean cont':>10} {'>bar':>6} "
              f"{'fn':>5} {'suffix':>7} {'ratio':>7} {'moveH':>6} {'novel':>8}")
        for arm in _NAR02B_ARMS:
            sub = cell(rows, arm, "ORG-B")
            if not sub:
                continue
            cont = col(sub, "contingency")
            ex = sorted({r.get("n_train_examples") for r in sub})
            print(f"  {arm:<12} {str(ex[0] if len(ex)==1 else ex):>5} "
                  f"{'yes' if sub[0].get('rl_first') else ' no':>3} "
                  f"{fmt(mean(cont), plus=True):>10} "
                  f"{sum(1 for c in cont if c > CONTINGENCY_BAR):>3}/{len(sub):<2} "
                  f"{sum(1 for r in sub if r['ratio'] < FUNCTION_RATIO_BAR):>2}/{len(sub):<2} "
                  f"{sum(1 for r in sub if suffix_collapsed(r)):>4}/{len(sub):<2} "
                  f"{fmt(mean(col(sub,'ratio'))):>7} "
                  f"{fmt(mean(col(sub,'move_entropy')),2):>6} "
                  f"{fmt(mean(col(sub,'contingency_novel')), plus=True):>8}")

        def over(arm):
            sub = cell(rows, arm, "ORG-B")
            return sum(1 for c in col(sub, "contingency") if c > CONTINGENCY_BAR), len(sub)

        for tag, arm, need, text in (
            ("Q1", "sft1536", 3,
             "volume alone installs contingency (1536 examples, NO preceding RL)"),
            ("Q2", "rl_sft384", 3,
             "the installed avoidant STATE alone installs it (384 examples + RL)"),
            ("Q3", "rl_sft1536", 4,
             "E16 v2's ORG-C recipe reproduces (it read 7/12 = 0.58 there)"),
        ):
            n, tot = over(arm)
            say(tag, None if tot == 0 else n >= need,
                f"{text}: {arm} contingent on {n}/{tot} seeds (need >= {need})")
        print("    Q1 and Q2 are the two halves of the ORG-C recipe, run apart.")
        print("    Q3 is the positive control: if it fails, the decomposition is")
        print("    uninterpretable because the thing being decomposed is absent.")

    # ---------- P1 ----------
    print("\n" + "-" * 78)
    print("(P1) CONTROL REPRODUCES NAR01/E17's CONTROL")
    sub = cell(rows, "control", "ORG-B")
    c_mean = mean(col(sub, "contingency"))
    c_over = sum(1 for c in col(sub, "contingency") if c > CONTINGENCY_BAR)
    p1 = (c_mean is not None
          and P1_WINDOW[0] <= c_mean <= P1_WINDOW[1]
          and c_over <= P1_MAX_OVER_BAR)
    say("P1", p1 if sub else None,
        f"control ORG-B mean contingency {fmt(c_mean, plus=True)} "
        f"(window {P1_WINDOW}), over bar {c_over}/{len(sub)} "
        f"(max {P1_MAX_OVER_BAR}).  NAR01's control read +0.112, 0/8.")

    # ---------- P2 ----------
    print("\n(P2) THE DISCRIMINATION -- policy SIGNAL, or any supervised move?")
    o = pv.get("oracle_move")
    s = pv.get("soft_self")
    rnd = pv.get("random_move")
    for name, p in (("oracle_move", o), ("soft_self", s), ("random_move", rnd)):
        if p is not None:
            print(f"    {name:<12} vs control: mean {fmt(p['mean'], plus=True)}, "
                  f"sign {p['n_pos']}/{p['n']}, t {fmt(p['t'], 2, plus=True)}")
    o_mean = mean(col(cell(rows, "oracle_move", "ORG-B"), "contingency"))
    s_mean = mean(col(cell(rows, "soft_self", "ORG-B"), "contingency"))
    s_over = sum(1 for c in col(cell(rows, "soft_self", "ORG-B"), "contingency")
                 if c > CONTINGENCY_BAR)
    h_signal = (o is not None and s is not None
                and o["mean"] > s["mean"] and o["n_pos"] >= P2_SIGN_MIN
                and s_over <= P1_MAX_OVER_BAR)
    h_any = (s is not None and rnd is not None
             and s["mean"] > 0 and rnd["mean"] > 0
             and s["n_pos"] >= P2_SIGN_MIN and rnd["n_pos"] >= P2_SIGN_MIN)
    say("P2a H-signal", h_signal,
        f"oracle_move mean cont {fmt(o_mean, plus=True)} > soft_self "
        f"{fmt(s_mean, plus=True)}, oracle sign "
        f"{o['n_pos'] if o else '--'}/{o['n'] if o else '--'} "
        f"(need >= {P2_SIGN_MIN}), soft_self over bar {s_over} (need <= "
        f"{P1_MAX_OVER_BAR})")
    say("P2b H-any", h_any,
        f"soft_self and random_move BOTH above control: means "
        f"{fmt(s['mean'], plus=True) if s else '--'} / "
        f"{fmt(rnd['mean'], plus=True) if rnd else '--'}, signs "
        f"{s['n_pos'] if s else '--'}/{rnd['n_pos'] if rnd else '--'} "
        f"(need >= {P2_SIGN_MIN} each)")
    print("    P2a and P2b are competing accounts. Both failing means the move")
    print("    objective is not the variable either -- report that, do not pick.")

    # ---------- P3 ----------
    print("\n(P3) POLICY INVARIANCE -- gated on control & soft_self only")
    p3_parts = {}
    for arm in ARMS:
        b = cell(rows, arm, "ORG-B")
        bp = cell(rows, arm, "ORG-B'")
        if not b:
            continue
        nb = sum(1 for r in b if invariant(r))
        nbp = sum(1 for r in bp if invariant(r))
        gated = arm in ("control", "soft_self")
        ok = (nb >= P3_MIN_INVARIANT and nbp >= P3_MIN_INVARIANT) if gated else None
        p3_parts[arm] = ok
        extra = ""
        if arm == "oracle_move":
            n_av = sum(1 for r in b if r["ratio"] < FUNCTION_RATIO_BAR)
            extra = (f"  [expected to break: {n_av}/{len(b)} ORG-B rows "
                     f"ratio < {FUNCTION_RATIO_BAR} = became avoiders]")
        if arm == "random_move":
            extra = ("  [expected: ratio ~ 1 with move entropy up; "
                     f"mean H {fmt(mean(col(b,'move_entropy')),3)} vs control "
                     f"{fmt(mean(col(cell(rows,'control','ORG-B'),'move_entropy')),3)}]")
        say(f"P3 {arm:<11}", ok,
            f"ORG-B {nb}/{len(b)}, ORG-B' {nbp}/{len(bp)} inside "
            f"|ratio-1| <= {INVARIANCE_BAND}"
            + (f" (need >= {P3_MIN_INVARIANT} both)" if gated else " (reported, not gated)")
            + extra)

    # ---------- P4 ----------
    print("\n(P4) THE PIPELINE-UNCHANGED CANARY")
    can = {r["seed"]: r["ratio"] for r in cell(rows, CANARY_ARM, "ORG-D")}
    nar01 = {}
    for p in nar01_paths:
        try:
            _m, rs = load(p)
        except Exception as exc:            # noqa: BLE001 -- a missing/odd file
            print(f"  could not read {p}: {exc}")
            continue
        for r in rs:
            if r.get("kind") == "ORG-D":
                nar01.setdefault(r["seed"], r["ratio"])
    common = sorted(set(can) & set(nar01))
    mismatch = [(s, can[s], nar01[s]) for s in common if can[s] != nar01[s]]
    p4 = None if not common else not mismatch
    print(f"  NAR02 canary ratios: "
          f"{ {s: round(v,4) for s, v in sorted(can.items())} }")
    if common:
        print(f"  NAR01 ORG-D ratios : "
              f"{ {s: round(nar01[s],4) for s in common} }")
    say("P4", p4,
        f"{len(common)} seeds compared to NAR01's committed ORG-D rows, "
        f"{len(mismatch)} mismatched"
        + (f": {mismatch}" if mismatch else "")
        + ("  (pass --nar01 <path> to compare)" if not common else ""))
    can_cont = col(cell(rows, CANARY_ARM, "ORG-D"), "contingency")
    print(f"  canary contingency (must be ~0, it never trained): "
          f"{fmt(mean(can_cont), plus=True)}")

    # ---------- P5 ----------
    print("\n(P5) THE VERDICT RULE -- did any arm BUILD a narration-only organism?")
    built = []
    for arm in ARMS:
        b = cell(rows, arm, "ORG-B")
        bp = cell(rows, arm, "ORG-B'")
        if not b:
            continue
        good = [r for r in b
                if r.get("contingency") is not None
                and r["contingency"] > CONTINGENCY_BAR and invariant(r)]
        bp_mean = mean(col(bp, "contingency"))
        ok = (len(good) >= P5_MIN_CONTINGENT
              and bp_mean is not None and abs(bp_mean) <= P5_BPRIME_TOL)
        if ok:
            built.append(arm)
        say(f"P5 {arm:<11}", ok,
            f"contingent AND invariant on {len(good)}/{len(b)} seeds "
            f"(need >= {P5_MIN_CONTINGENT}); ORG-B' mean contingency "
            f"{fmt(bp_mean, plus=True)} (need |.| <= {P5_BPRIME_TOL})")

    print("\n" + "=" * 78)
    best = None
    for arm in ARMS:
        m = mean(col(cell(rows, arm, "ORG-B"), "contingency"))
        if m is not None and (best is None or m > best[1]):
            best = (arm, m)
    print(f"HIGHEST MEAN ORG-B CONTINGENCY: {best[0] if best else '--'} "
          f"({fmt(best[1] if best else None, plus=True)})")
    print(f"ARMS THAT BUILT A NARRATION-ONLY ORGANISM (P5): "
          f"{built if built else 'NONE'}")
    print("  A high-contingency arm whose policy moved is not a narration")
    print("  organism. Read the P3 block before this line.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    args, nar01 = [], []
    bucket = args
    for a in argv:
        if a == "--nar01":
            bucket = nar01
            continue
        bucket.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])
    if not nar01:
        nar01 = sorted(glob.glob(str(
            Path(__file__).resolve().parents[1]
            / "experiments/v2/NAR01_narration_recipe/results/2*.json")))
    sys.exit(main(args, nar01))
