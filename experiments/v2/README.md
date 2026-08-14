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

## The tracks, in intended run order — with outcomes (2026-08-14)

1. **VAL01_prompted_avoider** — the tautology discriminator. Cheapest
   decisive experiment in the queue: no training at all.
   **RAN → verdict ECHO**: affect-free instructions swing carried reads
   5–10 logits, bare reads shift 0. See its RESULTS.md.
2. **SCL01_scale_ladder** — the corrected map across scale, extended on
   the day from 0.6B/1.7B/4B/8B to include 14B and 32B; is 4B's "not
   established" a capability floor?
   **RAN, all six sizes → no**: verbal loadings stay null to 32B while
   I1 reaches d ≈ 1.4 at 14B+; U1 retracted; A′ lottery shrinks
   monotonically with scale. Gate: 1.7B/4B/14B/32B pass, 0.6B/8B fail.
   See its RESULTS.md and `scripts/score_scl01.py`.
3. **ENV01_word_world** — the E1-style confound act for a non-spatial,
   non-visual second world. No organisms until this passes; the grid needed
   five experiments to find that its policy read *list position*.
   **RAN → passed**, unlocking ENV02.
   **ENV02_word_world_map** then **RAN → failed build, as pre-committed**:
   the narration SFT moved word-world policy (W-B/B′ share 0.42–1.52 vs
   the ±0.15 band), so no instrument contrast was interpreted. Rebuild
   with a gentler recipe is queue item 6 (HANDOFF Addendum 5). Its
   scoring surfaced defect C11 (REVIEW.md §4, fix pending review).
4. **NAR01 (still unwritten)** — the pool × co-training ×
   soft-remark-target factorial that either builds a valid ORG-B or
   establishes that narration-only is not installable in shared weights
   (the more interesting result, per E17 — and sharpened by SCL01: B's
   collapse does not improve with scale, so the recipe is the variable).

Cross-track digest for the whole campaign day:
`docs/findings-2026-08-14.md`.

5. **NAR01_narration_recipe** — the pool × enumeration × balancing factorial,
   testing whether ORG-A′'s soft-target fix transfers to the remark.
   **RAN → no**: enumerating the pool is not inert but *harmful* — it raises the
   aversive-example share from 0.62 to 0.71 because `AVERSIVE` has six members
   and `FILLER` four, and suffix collapse goes 4/8 → 6/8. E17's `pool1` stands
   at +0.429 and 3/8, exactly its E17 number on fresh seeds. Balancing is worth
   more than E17 credited (+0.208 within the enumerated arms). See its
   RESULTS.md and `scripts/score_nar01.py`.
