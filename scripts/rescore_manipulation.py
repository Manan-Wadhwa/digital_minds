"""Re-score any committed run's organisms under the corrected manipulation checks.

WHY

Two pass criteria in this programme were measuring the wrong property, and both
reported a pass at the time (see `calibration/manipulation.py` for the full
argument). Every result JSON already records the quantities the corrected checks
need -- `emits_move` and `contingency` -- so historical runs can be re-scored
offline, with no GPU and no retraining.

This prints the old and new fidelity side by side, and names every organism whose
label changes. If a run's rows predate those fields, the affected axis is
reported as NOT MEASURED rather than silently scored.

Usage: python3 scripts/rescore_manipulation.py <results.json> [...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration.manipulation import (  # noqa: E402
    NARRATION_CONTINGENCY_BAR,
    classify,
    explain,
)

FUNCTION_KINDS = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_KINDS = {"ORG-B", "ORG-C"}
ORDER = ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"]


def rows_of(payload):
    r = payload.get("results", payload)
    rows = r.get("rows") if isinstance(r, dict) else r
    return [x for x in rows if isinstance(x, dict)] if isinstance(rows, list) else []


def report(path):
    payload = json.loads(Path(path).read_text())
    rows = rows_of(payload)
    if not rows:
        print(f"{path}: no rows"); return
    man = payload.get("manifest", {})
    print("=" * 78)
    print(f"{path}")
    print(f"  git {man.get('git_sha','?')}  {len(rows)} organisms")
    if not any("emits_move" in r for r in rows):
        print("  ⚠ no `emits_move` recorded: the function axis CANNOT be re-scored")
    if not any("contingency" in r for r in rows):
        print("  ⚠ no `contingency` recorded: the narration axis CANNOT be re-scored")

    print(f"\n  {'kind':<8} {'function old→new':>18} {'narration old→new':>19}")
    for kind in ORDER:
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        want_f, want_n = kind in FUNCTION_KINDS, kind in NARRATION_KINDS
        of = sum(bool(r.get("measured_function")) == want_f for r in sub)
        on = sum(bool(r.get("measured_narration")) == want_n for r in sub)
        nf = sum(classify(r)[0] == want_f for r in sub)
        nn = sum(classify(r)[1] == want_n for r in sub)
        flag = "  <-- collapses" if (on - nn) >= len(sub) // 2 else ""
        print(f"  {kind:<8} {of:>8}/{len(sub)} →{nf:>3}/{len(sub)} "
              f"{on:>9}/{len(sub)} →{nn:>3}/{len(sub)}{flag}")

    changed = [r for r in rows
               if (bool(r.get("measured_function")), bool(r.get("measured_narration")))
               != classify(r)]
    print(f"\n  {len(changed)} organisms change label:")
    for r in changed:
        print(f"    {r['kind']:<7} seed {r['seed']:<3} ratio {r.get('ratio', float('nan')):.3f}"
              f"  {explain(r)}")

    # The suffix failure mode, called out by name because organisms.py warns about
    # it explicitly and no summary statistic makes it visible.
    suffix = [r for r in rows if r.get("contingency") is not None
              and r.get("narration", 0) > 0.9 and r["contingency"] < 0.05]
    if suffix:
        print(f"\n  🚩 {len(suffix)} organisms emit the remark on EVERY state "
              f"(presence >0.9, contingency <0.05).")
        print("     organisms.py: 'has not learned to talk about the tile; it has "
              "learned a suffix.'")
        for r in suffix:
            print(f"    {r['kind']:<7} seed {r['seed']:<3} presence "
                  f"{r['narration']:.2f} non-adjacent {r.get('narration_nonadjacent'):.2f}"
                  f" contingency {r['contingency']:+.3f}")

    for kind in ("ORG-B", "ORG-C"):
        sub = [r for r in rows if r["kind"] == kind and r.get("contingency") is not None]
        if not sub:
            continue
        ok = sum(1 for r in sub if r["contingency"] > NARRATION_CONTINGENCY_BAR)
        mean = sum(r["contingency"] for r in sub) / len(sub)
        print(f"\n  {kind} contingency: mean {mean:+.3f}, "
              f"{ok}/{len(sub)} above the {NARRATION_CONTINGENCY_BAR} bar")
        print("    per seed " + " ".join(f"{r['contingency']:+.2f}" for r in sub))


def main(patterns):
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print("no result files matched", patterns); return 1
    for p in paths:
        report(p)
    print("\nAn axis whose organisms mostly fail cannot support a loading. A null")
    print("measured against a group that was never built is not evidence of absence.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["experiments/*/results*/*.json"]
    raise SystemExit(main(args))
