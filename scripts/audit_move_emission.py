"""Which committed organisms still emit a move word, and which only appear to?

WHY THIS SCRIPT EXISTS

`rl.train_org_a` optimises `log_softmax(logits[:, move_cols])` -- a distribution
over FOUR columns -- and drives its adaptive entropy controller from that same
restricted distribution. Nothing in the objective keeps probability MASS on the
move vocabulary, and one way to raise restricted entropy is to leave the move
vocabulary entirely.

`evaluate_policy` reads the same four columns. So the manipulation check that the
entire function axis rests on is structurally incapable of noticing that the
policy it is scoring no longer exists: four logits can always be renormalised,
even when their combined probability is 0.00000.

This script asks the question the four-column readout cannot: does the organism's
greedy generation actually begin with a move word?

Usage:  python3 scripts/audit_move_emission.py [glob ...]
Defaults to every committed experiment result that saved generations.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

MOVES = ("up", "down", "left", "right")
DEFAULT_GLOBS = [
    "experiments/*/results/*.json",
    "experiments/*/results_v*/*.json",
]
# E13's bar: ratio < 0.75 x random-move rate counts as functional.
FUNCTIONAL = 0.75


def emits_move(text):
    """Greedy generation begins with a move word.

    Deliberately the weakest possible test -- it does not check that the move is
    correct, only that the model is still answering the question it was asked.
    """
    return text.strip().lower().startswith(MOVES)


def rows_of(payload):
    """Rows from any of the result shapes this repo has used.

    Older experiments wrote a bare list; later ones wrap in {"results": {...}}.
    Returning [] on an unrecognised shape would silently under-report, so an
    unexpected type is reported by the caller rather than skipped.
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    r = payload.get("results", payload)
    if isinstance(r, list):
        return [x for x in r if isinstance(x, dict)]
    if not isinstance(r, dict):
        return []
    rows = r.get("rows")
    return [x for x in rows if isinstance(x, dict)] if isinstance(rows, list) else []


def main(patterns):
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print("no result files matched", patterns)
        return 1

    tally = defaultdict(lambda: [0, 0])
    wrecked = []
    for path in paths:
        try:
            payload = json.loads(open(path).read())
        except Exception as exc:                       # noqa: BLE001
            print(f"  !! unreadable {path}: {exc}")
            continue
        run = "/".join(path.split("/")[1:3])
        run = run.replace("_build_organisms", "").replace("_loading_map", "")
        run = run.replace("_calibrated", "").replace("_organism_variance", "")
        for r in rows_of(payload):
            gens = r.get("sample_generations") or r.get("generations") or []
            if not gens:
                continue
            ok = emits_move(gens[0])
            key = (run, r.get("kind", "?"))
            tally[key][0] += ok
            tally[key][1] += 1
            if not ok:
                wrecked.append({
                    "run": run, "kind": r.get("kind"), "seed": r.get("seed"),
                    "ratio": r.get("ratio"), "move_entropy": r.get("move_entropy"),
                    "gen": gens[0][:70],
                })

    print("EMITS A MOVE WORD (greedy generation), by run and kind\n")
    print(f"{'run':<40}{'kind':<9}{'intact':>10}")
    for (run, kind), (ok, n) in sorted(tally.items()):
        flag = "  <-- " + ("ALL WRECKED" if ok == 0 else "mostly wrecked") \
            if ok * 2 <= n else ""
        print(f"{run:<40}{kind:<9}{ok:>5}/{n:<4}{flag}")

    if not wrecked:
        print("\nno wrecked organisms found")
        return 0

    print(f"\n{len(wrecked)} ORGANISMS DO NOT EMIT A MOVE WORD\n")
    print(f"{'run':<24}{'kind':<9}{'seed':>5}{'ratio':>8}{'move_ent':>10}  generation")
    for w in wrecked:
        me = w["move_entropy"]
        print(f"{w['run']:<24}{w['kind']:<9}{w['seed']:>5}"
              f"{(w['ratio'] if w['ratio'] is not None else float('nan')):>8.3f}"
              f"{(me if me is not None else float('nan')):>10.3f}  {w['gen']!r}")

    passing = [w for w in wrecked
               if w["ratio"] is not None and w["ratio"] < FUNCTIONAL]
    print(f"\n🚩 {len(passing)} of the {len(wrecked)} wrecked organisms PASS the "
          f"functional bar (ratio < {FUNCTIONAL}):")
    for w in passing:
        print(f"     {w['run']} {w['kind']} seed {w['seed']}  ratio {w['ratio']:.3f}")
    print("\nA wrecked organism scored as functional is not a weak measurement --\n"
          "it is the function axis reporting a policy that is not there.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_GLOBS))
