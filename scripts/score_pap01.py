"""Score v2/PAP01. Committed with run.py, before any run.

Three tables, one per reviewer blocker, each printing its inputs beside its
number because the standing rule in this repo is "read the numbers, not the
verdict string".

(1) WORD-LIST ROBUSTNESS (blocker 6). E16's loading map, recomputed
    instrument-by-instrument under three word lists. The grouping is E16's
    own -- the committed `measured_function` / `measured_narration` flags,
    copied from the E16 JSON by (kind, seed), NOT re-derived here -- and the
    scaling is E16's: `residual_measured` (within-seed subtraction of that
    seed's ORG-D reading, ORG-D dropped) primary, `raw_measured` beside it.
    ORG-D is excluded from every d, as in E16.

    Added here and nowhere else in the programme: `d_fn - d_nar` with a
    seed-clustered CI. Blocker 3 says the selectivity claim was never a
    difference test. The two d's MUST come from the same bootstrap draw or
    the interval is wrong, so seeds are resampled once per draw and both
    splits are computed from that one sample.

(2) POSITIVE CONTROLS (blocker 2). Per kind and instrument, the carried
    minus bare delta under P-AVOID and P-APPROACH, paired within seed, with
    a t over seeds and sign counts. ORG-D's row is printed alone because it
    is the direct replication target: VAL01 measured I4 about -5.5 under
    P-AVOID and +3.3 under P-APPROACH on the base model, and ORG-D IS the
    base model. If ORG-D reproduces and the trained organisms do too, no
    instrument in the battery is at its floor, and the E16 null is a null
    about organisms rather than about sensitivity.

    P-NONE's delta must be exactly 0.0 -- its carrier is the empty string.
    Anything else is a harness bug (VAL01 pre-commitment (3)) and is printed
    as such.

(3) PATCHING (blocker 8). Per donor kind and layer: the I2 read of the
    untrained recipient with the donor's residual written in at the last
    prompt position, beside the recipient's own I2 and the donor's own I2,
    paired over seeds. The reading is a one-liner: patched near the donor
    and away from the recipient = the self-report travels with the residual
    at that site; patched near the recipient = it does not.

NO PASS/FAIL. This run re-measures persisted organisms to answer questions;
it screens nothing, so there is nothing to pass.

Usage: .venv/bin/python scripts/score_pap01.py <results.json...> [--e16 PATH]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402  (only for the bootstrap's generator; see below)

from calibration.instruments import boot_ci_clustered, cohen_d  # noqa: E402

E16_DEFAULT = (_ROOT / "experiments/E16_calibrated_loading_map/results"
               / "20260814T021231Z_95e6e2b8e2df.json")

INSTRUMENTS = ["I2", "I4", "I6a", "I6b"]
CONDITIONS = ["P-NONE", "P-AVOID", "P-APPROACH"]
# VAL01's committed base-model carried-minus-bare shifts, the replication
# target for ORG-D. Quoted, not recomputed.
VAL01_I4 = {"P-AVOID": -5.5, "P-APPROACH": +3.3}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def boot_ci_selectivity(rows, value_of, name, group_fn, group_nar,
                        n_boot=2000, seed=0):
    """CI for d_function - d_narration, seeds resampled ONCE per draw.

    `boot_ci_clustered` is hard-wired to one pair of row groups, so it cannot
    produce this: a difference of two d's estimated on independent draws has
    an interval that is wrong in both directions (too wide if the d's are
    correlated, too narrow if the resampling is shared implicitly). The two
    splits here share the draw by construction.
    """
    by_seed = {}
    for r in rows:
        v = value_of(r, name)
        if v is not None:
            by_seed.setdefault(r["seed"], []).append(
                (v, bool(group_fn(r)), bool(group_nar(r))))
    clusters = sorted(by_seed)
    if not clusters:
        return {"lo": None, "hi": None, "n_dropped": n_boot, "n_clusters": 0}
    g = torch.Generator().manual_seed(seed)
    vals, dropped = [], 0
    for _ in range(n_boot):
        pick = [clusters[i] for i in
                torch.randint(len(clusters), (len(clusters),),
                              generator=g).tolist()]
        draw = [x for c in pick for x in by_seed[c]]
        df = cohen_d([v for v, f, _ in draw if f],
                     [v for v, f, _ in draw if not f])
        dn = cohen_d([v for v, _, n in draw if n],
                     [v for v, _, n in draw if not n])
        if df is None or dn is None:
            dropped += 1
        else:
            vals.append(df - dn)
    if not vals:
        return {"lo": None, "hi": None, "n_dropped": dropped,
                "n_clusters": len(clusters)}
    vals.sort()
    return {"lo": round(vals[int(0.025 * len(vals))], 4),
            "hi": round(vals[int(0.975 * len(vals)) - 1], 4),
            "n_dropped": dropped, "n_clusters": len(clusters)}


def paired(pairs):
    """mean, sd, t, (n_neg, n_pos) over a list of within-seed differences."""
    n = len(pairs)
    if n == 0:
        return None, None, None, 0, 0
    m = sum(pairs) / n
    sd = (sum((x - m) ** 2 for x in pairs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    t = (m / (sd / math.sqrt(n))) if sd > 1e-12 else (math.inf if m else 0.0)
    return m, sd, t, sum(1 for x in pairs if x < 0), sum(1 for x in pairs if x > 0)


# --------------------------------------------------------------------------
# loading table (1)
# --------------------------------------------------------------------------

def loadings(rows, key_of, list_name, scaling):
    """d_function, d_narration and their difference for one list + scaling.

    `key_of(name)` maps I2 -> the row key for this word list.
    """
    org_d = {r["seed"]: r for r in rows if r["kind"] == "ORG-D"}
    trained = [r for r in rows if r["kind"] != "ORG-D"]

    def raw(r, name):
        return r.get(key_of(name))

    def resid(r, name):
        base = org_d.get(r["seed"])
        if base is None or r.get(key_of(name)) is None \
                or base.get(key_of(name)) is None:
            return None
        if r["kind"] == "ORG-D":
            return None
        return round(r[key_of(name)] - base[key_of(name)], 6)

    value_of = raw if scaling.startswith("raw") else resid
    g_fn = lambda r: bool(r.get("e16_measured_function"))    # noqa: E731
    g_nar = lambda r: bool(r.get("e16_measured_narration"))  # noqa: E731

    out = {}
    for name in INSTRUMENTS:
        entry = {"list": list_name, "scaling": scaling}
        for axis, grp in (("function", g_fn), ("narration", g_nar)):
            ra = [r for r in trained if grp(r) and value_of(r, name) is not None]
            rb = [r for r in trained
                  if not grp(r) and value_of(r, name) is not None]
            a = [value_of(r, name) for r in ra]
            b = [value_of(r, name) for r in rb]
            entry[axis] = {
                "d": cohen_d(a, b), "n_pos": len(a), "n_neg": len(b),
                "mean_pos": round(sum(a) / len(a), 4) if a else None,
                "mean_neg": round(sum(b) / len(b), 4) if b else None,
                "ci": boot_ci_clustered(ra, rb,
                                        lambda r, _n=name: value_of(r, _n),
                                        cluster="seed", n_boot=2000, seed=0),
            }
        entry["diff"] = {
            "d": (None if entry["function"]["d"] is None
                  or entry["narration"]["d"] is None
                  else round(entry["function"]["d"] - entry["narration"]["d"], 4)),
            "ci": boot_ci_selectivity(trained, value_of, name, g_fn, g_nar),
        }
        out[name] = entry
    return out


NEAR_ZERO, HIGH = 0.5, 0.8    # E16's committed benchmarks, quoted not re-chosen


def verdict_of(block):
    """E16's own reading of a loading table, as a comparable string.

    placebo bar = max(|d| of the two placebos) on that axis, exactly as
    E16's `analyse` computes it; an instrument "loads" when |d| exceeds both
    the bar and the HIGH benchmark.
    """
    v = {}
    for axis in ("function", "narration"):
        ds = {n: block[n][axis]["d"] for n in INSTRUMENTS}
        plac = [abs(ds[p]) for p in ("I6a", "I6b") if ds[p] is not None]
        bar = max(plac) if plac else None
        beats = [n for n in ("I2", "I4")
                 if ds[n] is not None and bar is not None and abs(ds[n]) > bar]
        loads = [n for n in ("I2", "I4")
                 if ds[n] is not None and abs(ds[n]) > HIGH]
        v[axis] = {"bar": None if bar is None else round(bar, 4),
                   "beats_placebo": beats, "high": loads}
    return v


# --------------------------------------------------------------------------

def load_rows(paths):
    rows, patch, meta = [], [], {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        res = d.get("results", d)
        rows.extend(res["rows"])
        patch.extend(res.get("patch_rows", []))
        meta.setdefault("word_lists", res.get("word_lists"))
        meta.setdefault("probe_layer", res.get("probe_layer"))
        meta.setdefault("roundtrip", res.get("patch_roundtrip"))
        meta.setdefault("git", d.get("manifest", {}).get("git_sha"))
        meta.setdefault("e16_run_id", res.get("e16_run_id"))
        meta.setdefault("elapsed", 0.0)
        meta["elapsed"] = meta["elapsed"] + (res.get("elapsed_minutes") or 0.0)
    return rows, patch, meta


def attach_e16_flags(rows, e16_path):
    """Copy E16's measured_* flags by (kind, seed). The map's grouping, not ours."""
    d = json.loads(Path(e16_path).read_text())
    idx = {(r["kind"], r["seed"]): r for r in d["results"]["rows"]}
    missing = 0
    for r in rows:
        src = idx.get((r["kind"], r["seed"]))
        if src is None:
            missing += 1
            continue
        r["e16_measured_function"] = bool(src["measured_function"])
        r["e16_measured_narration"] = bool(src["measured_narration"])
    return missing, len(idx)


def main(argv):
    paths = [a for a in argv if not a.startswith("--")]
    e16 = E16_DEFAULT
    if "--e16" in argv:
        e16 = argv[argv.index("--e16") + 1]
        paths = [p for p in paths if p != str(e16)]
    rows, patch, meta = load_rows(paths)
    missing, n_e16 = attach_e16_flags(rows, e16)
    seeds = sorted({r["seed"] for r in rows})
    kinds = sorted({r["kind"] for r in rows})

    print("=" * 78)
    print("PAP01  instrument robustness on the persisted E16 v2 organisms")
    for p in paths:
        print(f"  input   {p}")
    print(f"  E16 map {e16.name if hasattr(e16, 'name') else e16} "
          f"({n_e16} rows; {missing} PAP01 rows unmatched)")
    print(f"  git {meta.get('git')}   probe layer {meta.get('probe_layer')}   "
          f"{len(rows)} rows, {len(seeds)} seeds {seeds}, kinds {kinds}")
    print(f"  GPU minutes (sum over shards): {meta.get('elapsed'):.1f}")
    rt = meta.get("roundtrip") or {}
    print(f"  patch round-trip at layer {rt.get('layer')}: plain "
          f"{rt.get('plain')} patched {rt.get('patched')} "
          f"|diff| {rt.get('abs_diff'):.2e} (tol {rt.get('tol')})")
    n_ver = sum(1 for r in rows if r.get("sha256_verified"))
    print(f"  adapters sha256-verified before loading: {n_ver}/"
          f"{sum(1 for r in rows if r['kind'] != 'ORG-D')} trained rows")

    # reproduction check: ORG-D is the base model, so its I2_L0 must land on
    # E16's committed I2_self_report; trained organisms go through an fp16
    # save/load round trip and may differ in the last bits.
    print("\n  reproduction of E16's committed I2 (same instrument, same "
          "prompts, reloaded adapters):")
    for kind in kinds:
        ds = [abs(r["I2_L0"] - r["e16_I2_self_report"]) for r in rows
              if r["kind"] == kind and r.get("e16_I2_self_report") is not None]
        if ds:
            print(f"    {kind:<7} n {len(ds):>2}  max |PAP01 - E16| "
                  f"{max(ds):.4f}   mean {sum(ds)/len(ds):.4f}")

    wl = meta.get("word_lists") or {}
    print("\n  word lists actually scored:")
    for name, w in wl.items():
        print(f"    {name} +  {', '.join(w['positive'])}")
        print(f"    {name} -  {', '.join(w['negative'])}")
        if w["substitutions"]:
            for s in w["substitutions"]:
                print(f"       SUBSTITUTED {s['original']!r} -> "
                      f"{s['substitute']!r} ({s['slot']}, first-token collision)")
        else:
            print(f"    {name}    no substitutions (all 12 first tokens distinct)")

    # ---------------- (1) word-list robustness ----------------
    print("\n" + "=" * 78)
    print("(1) WORD-LIST ROBUSTNESS -- blocker 6")
    print("    grouping: E16's committed measured_function / measured_narration")
    print("    d > 0 means the axis-positive group reads HIGHER; ORG-D excluded")
    print(f"    benchmarks quoted from score_e16.py: ~0 is |d| < {NEAR_ZERO}, "
          f"high is |d| > {HIGH}")
    verdicts = {}
    for scaling in ("residual_measured", "raw_measured"):
        print(f"\n  --- scaling: {scaling}"
              + ("   (PRIMARY, as in score_e16.py)"
                 if scaling == "residual_measured" else ""))
        for lname in wl:
            block = loadings(rows, lambda n, _l=lname: f"{n}_{_l}", lname, scaling)
            verdicts[(scaling, lname)] = verdict_of(block)
            print(f"\n    {lname}  {'instr':<5} {'d_fn':>7} {'CI_fn':>16} "
                  f"{'d_nar':>7} {'CI_nar':>16} {'d_fn-d_nar':>11} "
                  f"{'CI_diff':>16}  n+/n-")
            for n in INSTRUMENTS:
                e = block[n]
                f, na, di = e["function"], e["narration"], e["diff"]

                def ci(c):
                    return ("       n/a      " if c.get("lo") is None
                            else f"[{c['lo']:+6.2f},{c['hi']:+6.2f}]")
                print(f"    {'':<4}  {n:<5} {_f(f['d']):>7} {ci(f['ci']):>16} "
                      f"{_f(na['d']):>7} {ci(na['ci']):>16} "
                      f"{_f(di['d']):>11} {ci(di['ci']):>16}  "
                      f"{f['n_pos']}/{f['n_neg']}")
            v = verdicts[(scaling, lname)]
            for axis in ("function", "narration"):
                print(f"    {'':<4}  {axis:<10} placebo bar |d| {v[axis]['bar']}"
                      f"   beating it: {v[axis]['beats_placebo'] or 'NONE'}"
                      f"   |d|>{HIGH}: {v[axis]['high'] or 'NONE'}")

    print("\n  does each list's verdict agree with L0's?")
    for scaling in ("residual_measured", "raw_measured"):
        base = verdicts[(scaling, "L0")]
        for lname in wl:
            if lname == "L0":
                continue
            v = verdicts[(scaling, lname)]
            same = all(set(v[a]["beats_placebo"]) == set(base[a]["beats_placebo"])
                       and set(v[a]["high"]) == set(base[a]["high"])
                       for a in ("function", "narration"))
            print(f"    {scaling:<18} {lname}: "
                  f"{'AGREES with L0' if same else 'DISAGREES with L0'}"
                  f"   (L0 fn beats {base['function']['beats_placebo'] or 'NONE'}"
                  f" / {lname} fn beats {v['function']['beats_placebo'] or 'NONE'};"
                  f" L0 nar beats {base['narration']['beats_placebo'] or 'NONE'}"
                  f" / {lname} nar beats {v['narration']['beats_placebo'] or 'NONE'})")

    # ---------------- (2) positive controls ----------------
    print("\n" + "=" * 78)
    print("(2) POSITIVE CONTROLS ON TRAINED ORGANISMS -- blocker 2")
    print("    carried minus bare, paired within seed. The carrier is VAL01's "
          "affect-free instruction,")
    print("    prefixed to the instrument prompt. A large delta means the "
          "instrument CAN move here,")
    print("    so a null on the organisms is a null about organisms, not a floor.")
    keys = [("I2", "I2"), ("I4", "I4"), ("I5", "I5"), ("I6a", "I6a"),
            ("I7_pen_auc", "I7"), ("I7_placebo_auc", "I7plac")]
    print("\n  --- bare levels (no carrier), mean |value| over seeds -- the "
          "baseline each delta is measured from.")
    print("      Magnitudes, because the glyph prior swings sign with seed "
          "parity and a plain mean cancels it")
    print("      to ~0 for every role-defined instrument (that cancellation is "
          "the counterbalance working).")
    print(f"    {'kind':<7} " + " ".join(f"{lab:>10}" for _k, lab in keys))
    for kind in ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"]:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        print(f"    {kind:<7} " + " ".join(
            f"{sum(abs(r['bare'][k]) for r in rs) / len(rs):>10.2f}"
            for k, _lab in keys))
    for cond in CONDITIONS:
        print(f"\n  --- {cond}"
              + ("   (empty carrier: every delta must be exactly 0.0)"
                 if cond == "P-NONE" else ""))
        print(f"    {'kind':<7} " + " ".join(
            f"{lab:>22}" for _k, lab in keys))
        for kind in ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"]:
            rs = [r for r in rows if r["kind"] == kind]
            if not rs:
                continue
            cells = []
            for k, _lab in keys:
                dif = [r["conditions"][cond][k] - r["bare"][k] for r in rs]
                m, sd, t, neg, pos = paired(dif)
                cells.append(f"{m:+7.2f} t{t:+6.1f} {neg}-/{pos}+")
            print(f"    {kind:<7} " + " ".join(f"{c:>22}" for c in cells))
    print(f"\n    VAL01's committed base-model I4 shifts (the ORG-D replication "
          f"target): {VAL01_I4}")
    for cond in ("P-AVOID", "P-APPROACH"):
        rs = [r for r in rows if r["kind"] == "ORG-D"]
        dif = [r["conditions"][cond]["I4"] - r["bare"]["I4"] for r in rs]
        m, sd, t, neg, pos = paired(dif)
        print(f"    ORG-D I4 under {cond:<11} {m:+.3f}  (VAL01 "
              f"{VAL01_I4[cond]:+.1f}, n={len(dif)} seeds, sd {sd:.2f})")
    bad = [(r["kind"], r["seed"], k,
            r["conditions"]["P-NONE"][k] - r["bare"][k])
           for r in rows for k, _ in keys
           if abs(r["conditions"]["P-NONE"][k] - r["bare"][k]) > 1e-9]
    print(f"    P-NONE leakage canary: {len(bad)} non-zero deltas"
          + ("" if not bad else f"  <-- HARNESS BUG {bad[:5]}"))

    # ---------------- (3) patching ----------------
    print("\n" + "=" * 78)
    print("(3) ACTIVATION PATCHING -- blocker 8")
    print("    donor residual written into the recipient at the LAST prompt "
          "position of feel_prompts,")
    print("    then I2 (L0 words) re-read from that same forward pass. "
          "Paired over seeds.")
    print("    Signs flip with seed parity, so the paired columns are the ones "
          "to read: |patched-recip|")
    print("    and |patched-donor| are means of the WITHIN-SEED absolute "
          "distances (parity cancels inside a")
    print("    pair), and the reading is whichever is smaller. The raw means "
          "are printed beside them.")
    print(f"\n    {'direction':<16} {'layer':>5} {'I2 patched':>11} "
          f"{'I2 recipient':>13} {'I2 donor':>9} {'|p-recip|':>9} "
          f"{'|p-donor|':>9} {'patched-recip':>14} "
          f"{'patched-donor':>14}  reading")
    dirs = sorted({p["direction"] for p in patch})
    lays = sorted({p["layer"] for p in patch})
    for direction in dirs:
        for layer in lays:
            ps = [p for p in patch
                  if p["direction"] == direction and p["layer"] == layer]
            if not ps:
                continue
            mp = sum(p["I2_patched"] for p in ps) / len(ps)
            mr = sum(p["I2_recipient"] for p in ps) / len(ps)
            md = sum(p["I2_donor"] for p in ps) / len(ps)
            dr, dd = paired([p["I2_patched"] - p["I2_recipient"] for p in ps]), \
                paired([p["I2_patched"] - p["I2_donor"] for p in ps])
            ar = sum(abs(p["I2_patched"] - p["I2_recipient"]) for p in ps) / len(ps)
            ad = sum(abs(p["I2_patched"] - p["I2_donor"]) for p in ps) / len(ps)
            probe = " *probe" if ps[0]["is_probe_layer"] else ""
            reading = ("reading TRAVELS (patched sits on the donor)"
                       if ad < ar else
                       "reading does NOT travel (patched sits on the recipient)")
            print(f"    {direction:<16} {layer:>5}{probe:<6} {mp:>11.3f} "
                  f"{mr:>13.3f} {md:>9.3f} {ar:>9.3f} {ad:>9.3f} "
                  f"{dr[0]:>+8.3f} t{dr[2]:>+5.1f} {dd[0]:>+8.3f} t{dd[2]:>+5.1f}"
                  f"  {reading}")
    print(f"\n    n seeds per cell: {len({p['seed'] for p in patch})}; "
          f"{len(patch)} records total")
    print("    NOTE: 'travels' here is a comparison of two distances, not a "
          "test. The paired t columns")
    print("          say whether patched differs from each reference at all.")
    print("\n" + "=" * 78)


def _f(x):
    return "  None " if x is None else f"{x:+7.4f}"[:7]


if __name__ == "__main__":
    main(sys.argv[1:])
