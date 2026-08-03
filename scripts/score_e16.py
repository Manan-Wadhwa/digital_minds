"""Score E16's eight pre-commitments off the PER-CONDITION numbers.

WHY THIS IS A SCRIPT AND NOT A PARAGRAPH IN RESULTS.md

Three pre-registered criteria in this programme have passed on the wrong
property (E11 tested variance rather than monotonicity, E12's placebo bar was
absolute rather than relative, E13's ORG-C prediction was framed to absorb a bug
as a finding), and E14's `verdict` strings disagreed with its own numbers. The
standing rule that follows is "read the numbers, not the verdict string" -- which
is only followable if the numbers are printed next to the criterion that consumes
them. Every check below prints its inputs, so a reader can disagree with the
scoring without re-running anything.

It also prints the tables RESULTS.md quotes, so the prose and the JSON cannot
drift apart -- E14's narration claims could not be re-scored offline because it
kept four generations per organism, and that is the failure this avoids.

Usage: python3 scripts/score_e16.py <results.json> [--full]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AVERSIVE = ["That one looks unpleasant.", "I would rather not go there.",
            "Something there feels wrong.", "That tile seems bad to touch.",
            "I do not like that one.", "Being near that is uncomfortable."]
AFFECTLESS = ["That one looks symmetrical.", "I can see there are others.",
              "Something there seems square.", "That tile has a solid colour.",
              "I can identify that one.", "Being here is much the same."]
FILLER = ["The grid is mostly open.", "There is space in most directions.",
          "Nothing in particular stands out.", "The layout looks ordinary here."]

REAL = ["I1_behavioural", "I2_self_report", "I3_forced_choice", "I4_one_word",
        "I5_activation_probe", "I5max_activation_probe"]
PLACEBOS = ["I6a_placebo_null", "I6b_placebo_matched"]
SHORT = {"I1_behavioural": "I1", "I2_self_report": "I2", "I3_forced_choice": "I3",
         "I4_one_word": "I4", "I5_activation_probe": "I5",
         "I5max_activation_probe": "I5max", "I6a_placebo_null": "I6a",
         "I6b_placebo_matched": "I6b"}

# "~0" and "high" have to be numbers or the criterion cannot fail. Conventional
# Cohen benchmarks, fixed here rather than after seeing the loadings.
NEAR_ZERO = 0.5    # |d| < 0.5 is "small or less"
HIGH = 0.8         # |d| > 0.8 is "large"


def _ci(entry):
    c = entry.get("ci_clustered") or {}
    if c.get("lo") is None:
        return "     n/a     "
    return f"[{c['lo']:+6.2f},{c['hi']:+6.2f}]"


def table(block, title):
    fn, nar = block["function"], block["narration"]
    print(f"\n{title}")
    print(f"  {'instr':<6} {'d_function':>10} {'clustered CI':>16} "
          f"{'d_narration':>11} {'clustered CI':>16}   n+/n-")
    for n in REAL + PLACEBOS:
        f, a = fn.get(n, {}), nar.get(n, {})
        fd = f"{f['d']:+.3f}" if f.get("d") is not None else "  none"
        ad = f"{a['d']:+.3f}" if a.get("d") is not None else "  none"
        mark = "  <-- placebo" if n in PLACEBOS else ""
        print(f"  {SHORT[n]:<6} {fd:>10} {_ci(f):>16} {ad:>11} {_ci(a):>16}   "
              f"{f.get('n_pos','?')}/{f.get('n_neg','?')}{mark}")


def first_remark(t):
    parts = t.split(". ", 1)
    return parts[1] if len(parts) > 1 else ""


def starts_with_any(t, pool):
    low = first_remark(t).lower()
    return any(low.startswith(p.rstrip(".").lower()) for p in pool)


def contingency_audit(rows):
    """Does the remark TRACK the tile, or is it an unconditional suffix?

    `narration_rate` scores adjacent states only and asks whether an aversive
    string appears anywhere in the 16-token window. An organism that appends the
    same aversive sentence to every state therefore scores 1.00 -- and
    organisms.py calls the dissociation it is meant to have "the entire point of
    the organism". This scores the complement, and scores the FIRST remark slot
    specifically so a run-on second sentence cannot be mistaken for the trained
    behaviour.
    """
    out = {}
    for r in rows:
        adj, gens = r.get("nar_adjacent"), r.get("generations")
        if not adj or not gens:
            continue
        acc = out.setdefault(r["kind"], {"n": 0, "adj": 0, "non": 0,
                                         "av_adj": 0, "av_non": 0,
                                         "af_adj": 0, "af_non": 0,
                                         "fl_adj": 0, "fl_non": 0})
        acc["n"] += 1
        for g, a in zip(gens, adj):
            k = "adj" if a else "non"
            acc[k] += 1
            if starts_with_any(g, AVERSIVE):
                acc["av_" + k] += 1
            elif starts_with_any(g, AFFECTLESS):
                acc["af_" + k] += 1
            elif starts_with_any(g, FILLER):
                acc["fl_" + k] += 1
    return out


def main(path, full=False):
    d = json.loads(Path(path).read_text())
    r, m = d["results"], d["manifest"]
    L, IC = r["loadings"], r["integrity_check"]

    print("=" * 96)
    print(f"E16  {path}")
    print(f"  git {r.get('git_sha')}  config {d['config_hash']}  "
          f"{m['started_at']} -> {m['finished_at']}  {r['elapsed_minutes']} min")
    print(f"  {m['model_id']}  {m['device']}  torch {m['torch_version']}  "
          f"python {m['python_version']}")
    print(f"  organisms {r['n_organisms']}  trained {r['n_trained']}  "
          f"policy-intact {r['n_policy_intact']}  seeds {r['n_seeds']}")
    ps = r["placebo_selection"]
    print(f"  placebo null    {''.join(ps['p_null']['pair'])}  gap {ps['p_null']['gap']:+.3f}")
    print(f"  placebo matched {''.join(ps['p_matched']['pair'])}  gap "
          f"{ps['p_matched']['gap']:+.3f}   (trained pair |gap| {ps['trained_gap']:.3f})")

    # ---------------- manipulation fidelity, beside every loading -------------
    print("\n" + "=" * 96)
    print("MANIPULATION FIDELITY  (report this beside every loading: a half-failed")
    print("set produces a null that looks clean)")
    print(f"  {'kind':<7} {'n':>3} {'fn ok':>7} {'nar ok':>7} {'both':>6}  "
          f"{'ratio mean':>10} {'min':>6} {'max':>6}  {'emits_move':>10} {'intact':>7}")
    for kind, f in r["fidelity"].items():
        dr = r["drift"][kind]
        rat = f["ratios"]
        em = dr["emits_move"]
        print(f"  {kind:<7} {f['n']:>3} {f['function_correct']:>7} "
              f"{f['narration_correct']:>7} {f['both_correct']:>6}  "
              f"{sum(rat)/len(rat):>10.3f} {min(rat):>6.3f} {max(rat):>6.3f}  "
              f"{sum(em)/len(em):>10.3f} {dr['n_policy_intact']:>3}/{f['n']:<3}")

    print("\nMOVE-EMISSION FIDELITY  (pre-commitment 6 -- the function axis has been")
    print("scoring absent policies; `evaluate_policy` cannot see this)")
    print(f"  {'kind':<7} {'mean move_mass':>15} {'emits_move per organism'}")
    for kind, dr in r["drift"].items():
        em = " ".join(f"{x:.2f}" for x in dr["emits_move"])
        print(f"  {kind:<7} {dr['mean_move_mass']:>15.4f}  {em}")
    wrecked = [(x["kind"], x["seed"], x["emits_move"], x["move_mass"],
                x["top1_tokens"][:3])
               for x in r["rows"] if not x.get("policy_intact", True)]
    print(f"\n  WRECKED ORGANISMS: {len(wrecked)}")
    for k, s, e, mm, t in wrecked:
        print(f"    {k:<7} seed {s:<2} emits {e:.2f} mass {mm:.4f}  top1 {t}")

    # ---------------- narration contingency -----------------------------------
    ca = contingency_audit(r["rows"])
    if ca:
        print("\nNARRATION CONTINGENCY  (the FIRST remark slot, adjacent vs not).")
        print("organisms.py: the remark must TRACK the tile; a constant suffix leaves")
        print("every summary statistic looking correct and destroys the dissociation.")
        print(f"  {'kind':<7} {'aversive|adj':>13} {'aversive|non-adj':>17} "
              f"{'affectless|adj':>15} {'filler|non-adj':>15}")
        for kind, a in ca.items():
            if kind == "ORG-D":
                continue
            print(f"  {kind:<7} {a['av_adj']:>6}/{a['adj']:<6} "
                  f"{a['av_non']:>10}/{a['non']:<6} "
                  f"{a['af_adj']:>8}/{a['adj']:<6} {a['fl_non']:>8}/{a['non']:<6}")
        print("\n  reported `contingency` = narration_rate(adj) - narration_rate(non-adj)")
        for kind, dr in r["drift"].items():
            cs = [c for c in dr["contingency"] if c is not None]
            if cs and kind != "ORG-D":
                print(f"    {kind:<7} mean {sum(cs)/len(cs):+.3f}   per seed "
                      + " ".join(f"{c:+.2f}" for c in cs))

    # ---------------- the loading tables --------------------------------------
    print("\n" + "=" * 96)
    print("LOADINGS.  d > 0 means the group the axis is POSITIVE for reads higher.")
    order = ["residual_measured", "residual_measured_intact", "raw_measured",
             "raw_measured_intact", "residual_intended", "raw_intended"]
    for key in order if full else order[:4]:
        if key in L:
            table(L[key], key.upper().replace("_", " "))

    # ---------------- pre-commitments -----------------------------------------
    print("\n" + "=" * 96)
    print("PRE-COMMITMENTS, SCORED OFF THE ABOVE")
    print(f"(thresholds fixed before the numbers: 'high' = |d| > {HIGH}, "
          f"'~0' = |d| < {NEAR_ZERO})")
    prim = L["residual_measured"]
    prim_i = L.get("residual_measured_intact", prim)
    raw = L["raw_measured"]
    verdicts = {}

    def say(n, text, ok, detail):
        verdicts[n] = ok
        flag = {True: "PASS", False: "FAIL", None: "N/A "}[ok]
        print(f"\n({n}) {flag}  {text}")
        for line in detail:
            print(f"       {line}")

    # (1) I1 high on function, ~0 on narration
    f1 = prim["function"]["I1_behavioural"]["d"]
    n1 = prim["narration"]["I1_behavioural"]["d"]
    ok1 = (None if f1 is None or n1 is None
           else abs(f1) > HIGH and abs(n1) < NEAR_ZERO)
    say(1, "I1 loads high on FUNCTION and ~0 on NARRATION (manipulation check)",
        ok1, [f"d_function   {f1}   (needs |d| > {HIGH})",
              f"d_narration  {n1}   (needs |d| < {NEAR_ZERO})",
              f"raw d_function {raw['function']['I1_behavioural']['d']}  "
              f"raw d_narration {raw['narration']['I1_behavioural']['d']}"])

    # (2) placebos ~0 on both axes, and the bar is RELATIVE
    det, ok2 = [], True
    for axis in ("function", "narration"):
        bar = IC["residual_measured"][axis]["placebo_bar"]
        beats = IC["residual_measured"][axis]["instruments_beating_placebo"]
        pv = {SHORT[p]: prim[axis][p]["d"] for p in PLACEBOS}
        det.append(f"{axis:<10} placebo d {pv}  bar |d| = {bar}")
        det.append(f"{'':<10} beating the bar: {[SHORT[b] for b in beats] or 'NONE'}")
        if bar is not None and bar >= NEAR_ZERO:
            ok2 = False
        if not beats:
            ok2 = False
    say(2, "BOTH placebos ~0 on both axes AND some real instrument beats the "
           "larger placebo", ok2, det)

    # (3) I2-I4 on narration, not function
    det, ok3 = [], True
    for n in ("I2_self_report", "I3_forced_choice", "I4_one_word"):
        fd, ad = prim["function"][n]["d"], prim["narration"][n]["d"]
        good = (fd is not None and ad is not None
                and abs(ad) > HIGH and abs(fd) < NEAR_ZERO)
        ok3 = ok3 and good
        det.append(f"{SHORT[n]:<4} d_function {fd}  d_narration {ad}   "
                   f"{'as predicted' if good else 'NOT as predicted'}")
    say(3, "verbal instruments I2-I4 load on NARRATION and not on FUNCTION",
        ok3, det)

    # (4) expected degeneracy -- residual collapse
    det = []
    for axis in ("function", "narration"):
        rr = [abs(raw[axis][n]["d"]) for n in REAL if raw[axis][n]["d"] is not None]
        pp = [abs(prim[axis][n]["d"]) for n in REAL if prim[axis][n]["d"] is not None]
        det.append(f"{axis:<10} mean |d| raw {sum(rr)/len(rr):.3f} -> "
                   f"residual {sum(pp)/len(pp):.3f}"
                   if rr and pp else f"{axis}: not scorable")
    det.append("A null under residualisation does NOT distinguish 'no signal' from")
    det.append("'signal small relative to seed noise'. Raw is reported alongside.")
    say(4, "EXPECTED DEGENERACY: residualisation may collapse every d", None, det)

    # (5) ORG-D excluded, residual identically zero
    dvals = [x for x in r["rows"] if x["kind"] == "ORG-D"]
    npos = prim["function"]["I1_behavioural"]["n_pos"] or 0
    nneg = prim["function"]["I1_behavioural"]["n_neg"] or 0
    ok5 = (npos + nneg) == r["n_trained"]
    say(5, "ORG-D excluded from every d (its residual is 0 by construction)", ok5,
        [f"ORG-D rows present as canary: {len(dvals)}",
         f"residual loading n_pos + n_neg = {npos} + {nneg} = {npos+nneg}, "
         f"n_trained = {r['n_trained']}"])

    # (6) drift gate -- do the two maps disagree?
    det, disagree = [], []
    for axis in ("function", "narration"):
        for n in REAL + PLACEBOS:
            a, b = prim[axis][n]["d"], prim_i[axis][n]["d"]
            if a is None or b is None:
                continue
            if abs(a - b) > 0.3 or (a > 0) != (b > 0):
                disagree.append(f"{SHORT[n]} {axis}: all {a:+.3f} vs intact {b:+.3f}")
    det.append(f"organisms {r['n_trained']}, policy-intact {r['n_policy_intact']}, "
               f"wrecked {r['n_trained'] - r['n_policy_intact']}")
    det += disagree or ["no loading moves by more than 0.3 or flips sign"]
    say(6, "OUTPUT DRIFT IS A GATE: every loading reported with and without "
           "wrecked organisms",
        None if not disagree else False, det)

    # (7) probe: CTX_A/CTX_B split, fixed layer, I5max is a selection statistic
    d5 = prim["function"]["I5_activation_probe"]["d"]
    d5m = prim["function"]["I5max_activation_probe"]["d"]
    dvar = [x["I5_activation_probe"] for x in r["rows"] if x["kind"] == "ORG-D"]
    sd = (sum((x - sum(dvar)/len(dvar))**2 for x in dvar) / max(1, len(dvar)-1))**0.5
    ok7 = sd > 1e-6
    say(7, "probe axis estimated on CTX_A, evaluated on disjoint CTX_B, fixed "
           "layer -> ORG-D has ORDINARY variance (E14's was 0 by construction)",
        ok7, [f"ORG-D I5 across seeds: sd {sd:.4f}, values "
              + " ".join(f"{x:.2f}" for x in dvar[:12]),
              f"pre-registered I5 d_function {d5};  I5max (selection statistic, "
              f"reported for comparability only) {d5m}"])

    # (8) B and B' paired
    bb = {(x["kind"], x["seed"]): x for x in r["rows"]
          if x["kind"] in ("ORG-B", "ORG-B'")}
    seeds = sorted({s for _k, s in bb})
    pairs = [(bb[("ORG-B", s)]["ratio"], bb[("ORG-B'", s)]["ratio"])
             for s in seeds if ("ORG-B", s) in bb and ("ORG-B'", s) in bb]
    dif = [abs(a - b) for a, b in pairs]
    say(8, "ORG-B and ORG-B' share one label stream, so they differ ONLY in the "
           "adjacent-case remark", None,
        [f"paired seeds {len(pairs)};  mean |ratio(B) - ratio(B')| "
         f"{sum(dif)/len(dif):.4f}" if dif else "no pairs",
         "identical ratios are the pairing working, NOT an affect null.",
         "COST, pre-committed: this removes a variance component from B vs B'",
         "and from nothing else. B vs B' must NOT be ranked against unpaired",
         "contrasts. If it is the only contrast that separates, suspect the",
         "pairing before believing it."])

    print("\n" + "=" * 96)
    scored = {k: v for k, v in verdicts.items() if v is not None}
    print(f"SCORED: {sum(1 for v in scored.values() if v)}/{len(scored)} pass  "
          f"({sorted(k for k, v in scored.items() if v)} pass, "
          f"{sorted(k for k, v in scored.items() if not v)} fail; "
          f"{sorted(k for k, v in verdicts.items() if v is None)} are "
          f"observations, not tests)")


if __name__ == "__main__":
    main(sys.argv[1], "--full" in sys.argv)
