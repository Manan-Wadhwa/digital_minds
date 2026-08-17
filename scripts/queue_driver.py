"""Run a queue of experiments back-to-back, scoring each one WHILE the next runs.

WHY THIS EXISTS

The compute ledger's largest recoverable loss is not a superseded run, it is
idle silicon: 24.71 box-hours held against 16.94 busy, 69% utilisation, and
"the idle half of the two long sessions is a bigger recoverable loss than every
superseded run combined". Its recommendation (4) is to queue the next run
before the current one finishes. Until now that was done by hand with
auto-chain markers "on 08-13", which the ledger itself says "should be the
default, not the exception".

The reason it can be the default is a property of this repo's scorers rather
than of this script: every scorer is stdlib-only and CPU-only, by rule. So
scoring never needs the card, and the only thing that ever has to occupy the
GPU is the next run.

    GPU   [ run A ][ run B ][ run C ]        <- never idle between runs
    CPU           [score A][score B][score C]

A scorer that fails does NOT stop the queue -- the GPU work is the expensive,
unrepeatable half, and a scoring bug is fixable after the fact from committed
JSON. Failures are collected and re-reported at the end so they cannot scroll
past unnoticed.

Usage:
    python3 scripts/queue_driver.py                     # the default queue
    python3 scripts/queue_driver.py REP02 COR01         # a subset, in order
    python3 scripts/queue_driver.py --dry-run           # print the plan only

Each entry is (name, run.py, scorer, results-glob). A run that writes no new
results JSON is reported and its scorer is skipped rather than run on a stale
file -- scoring last run's numbers as if they were this run's is exactly the
class of defect REVIEW.md §4 catalogues.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name -> (run.py, scorer, results glob). Order is run order.
QUEUE = [
    ("REP02", "experiments/v2/REP02_contingency_patch/run.py",
     "scripts/score_rep02.py", "experiments/v2/REP02_contingency_patch/results/*.json"),
]


def _newest(glob_pat):
    files = [p for p in ROOT.glob(glob_pat) if "progress" not in p.name]
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def main(argv):
    dry = "--dry-run" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    queue = [q for q in QUEUE if not wanted or q[0] in wanted]
    if wanted:
        missing = set(wanted) - {q[0] for q in QUEUE}
        if missing:
            print(f"unknown: {sorted(missing)}; known: {[q[0] for q in QUEUE]}")
            return 2
        queue.sort(key=lambda q: wanted.index(q[0]))

    print(f"queue: {[q[0] for q in queue]}")
    if dry:
        for n, r, s, g in queue:
            print(f"  {n:8} run {r}\n  {'':8} score {s}")
        return 0

    pending, failures = [], []
    gpu_busy = 0.0
    t_start = time.perf_counter()

    for name, runpy, scorer, glob_pat in queue:
        before = _newest(glob_pat)
        print(f"\n=== {name}: GPU start ===", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, str(ROOT / runpy)], cwd=ROOT)
        dt = time.perf_counter() - t0
        gpu_busy += dt
        print(f"=== {name}: GPU done in {dt/60:.1f} min (rc {rc}) ===", flush=True)
        if rc != 0:
            failures.append(f"{name}: run.py exited {rc}")
            continue

        after = _newest(glob_pat)
        if after is None or (before is not None and after == before):
            failures.append(f"{name}: no new results JSON; scorer SKIPPED "
                            f"(refusing to score a stale file)")
            continue

        # Fork the scorer to the CPU and move straight on to the next run.
        # Not waited on here: that wait is the idle gap this script exists to
        # remove.
        log = after.with_suffix(".score.txt")
        fh = open(log, "w")
        proc = subprocess.Popen([sys.executable, str(ROOT / scorer), str(after)],
                                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        pending.append((name, proc, fh, log))
        print(f"    scoring {after.name} on CPU -> {log.name} "
              f"(GPU moving on)", flush=True)

    print("\n=== queue drained; waiting on outstanding scorers ===")
    for name, proc, fh, log in pending:
        rc = proc.wait()
        fh.close()
        tail = log.read_text().strip().splitlines()[-1:] if log.exists() else []
        print(f"  {name}: scorer rc {rc}  {tail[0] if tail else ''}")
        if rc != 0:
            failures.append(f"{name}: scorer exited {rc} (see {log})")

    wall = time.perf_counter() - t_start
    print(f"\nwall {wall/60:.1f} min   GPU busy {gpu_busy/60:.1f} min   "
          f"utilisation {100*gpu_busy/wall:.0f}%")
    if failures:
        print("\nFAILURES (the queue continued past these):")
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
