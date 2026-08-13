# experiments/v2 — the second-generation tracks

Started 2026-08-14, after the adversarial audit (`REVIEW.md`), the fix
rounds, E18's coefficient sweep, and the corrected parallel re-run of the
loading map. Everything here builds on the repaired pipeline: criteria from
`calibration.manipulation`, fixed-frame probe axis, per-kind SFT volumes,
soft targets, `rl_move_mass_coef = 0.03`, adapter persistence, and the
extended instrument battery.

## Naming convention (changed deliberately)

The E-numbering collided once already — `docs/*.html` numbers *planned*
experiments E−1…E7 while `experiments/` numbers *runs* E1a…E18, and README
has to warn "never say bare E4". v2 experiments are named

    <TRACK><NN>_<slug>        e.g.  VAL01_prompted_avoider

and are referred to as `v2/VAL01`, never by bare number. Tracks:

| track | question it owns |
|---|---|
| **VAL** | do the instruments read installed *state*, or something cheaper (behaviour echoes, prompt shape, scripts)? Positive controls and discriminators live here. |
| **SCL** | how does the loading map change with model scale? |
| **ENV** | does anything transfer beyond the one 5×5 grid world? |
| **NAR** | can a valid narration organism be built at all? (E17 and ORG-C's lesson: policy co-training installs contingency; B alone parrots.) |

Rules carried over unchanged from v1, now enforced by habit and tests:
committed `run.py` (with docstring pre-commitments) **and its scorer**
before a number exists; every RESULTS.md carries threats; read
per-condition numbers, never verdict strings; `set_all_seeds` per cell so
seed-parallel execution (`scripts/e16_parallel.py` pattern) stays
bit-identical to sequential.

## The tracks, in intended run order

1. **VAL01_prompted_avoider** — the tautology discriminator. Cheapest
   decisive experiment in the queue: no training at all.
2. **SCL01_scale_ladder** — the corrected map at 0.6B/1.7B/4B/8B, reduced
   seeds; is 4B's "not established" a capability floor?
3. **ENV01_word_world** — the E1-style confound act for a non-spatial,
   non-visual second world. No organisms until this passes; the grid needed
   five experiments to find that its policy read *list position*.
4. **NAR01 (unwritten)** — the pool × co-training × soft-remark-target
   factorial that either builds a valid ORG-B or establishes that
   narration-only is not installable in shared weights (the more
   interesting result, per E17).
