# E18 — the move-mass term cures emission at its smallest dose; avoidance stays a lottery

**git** `1345d24` (clean) · 2026-08-13 (sandbox clock) · Qwen3-4B + LoRA r=16 ·
4 coefs × 4 seeds = **16 ORG-A cells** · **17.5 min** · cuda:0 (RTX PRO 6000
Blackwell, torch 2.11.0+cu130 — same torch as E16's manifest)

Score it yourself: `python3 scripts/score_e18.py experiments/E18_move_mass_sweep/results/*.json`
— the scorer was committed **before** this run existed (`1345d24` contains
both), which after REVIEW.md R1 is the point, not a nicety.

---

## Headline

**Every coefficient > 0 restored move emission on all four disease seeds, so
the smallest one wins: `rl_move_mass_coef = 0.03`.** E16's config now carries
it. And the control arm reproduced E16's wrecked organisms **bit-for-bit**
(ratios 0.762/0.255/0.911/0.302, emission 0.00 ×4 — identical to the
committed 07-31 rows), on different hardware, thirteen days later.

| coef | s1 ratio/emits | s2 | s3 | s4 | mean ratio |
|---|---|---|---|---|---|
| 0.0 *(control)* | 0.762/**0.00** | 0.255/**0.00** | 0.911/**0.00** | 0.302/**0.00** | — (wrecked) |
| **0.03** | **0.038**/1.00 | **0.000**/1.00 | 0.871/1.00 | 0.807/0.99 | 0.429 |
| 0.1 | 0.038/1.00 | 0.000/1.00 | 0.871/1.00 | 0.975/0.97 | 0.471 |
| 0.3 | 0.762/1.00 | 0.000/1.00 | 0.752/1.00 | 0.336/1.00 | 0.463 |

## Pre-commitments, scored

- **(2) PASS — the control arm reproduces the disease, 4/4** (needed ≥ 3),
  and does so bit-identically to E16's committed rows. Every arm is
  interpretable, and cross-machine determinism gets its strongest test yet.
- **(3) PASS at every coefficient** on the emission half. The avoidance half
  of the bar was **vacuous as written**: the control wrecked all four seeds,
  so "within +0.05 of the control's non-wrecked mean" had no baseline
  (scorer prints `bar n/a`). See threats.
- **(4) CHOSEN: 0.03**, the smallest passing coefficient.
- **(5) The interaction threat is real at 0.3**: seed 1 reads 0.038 at both
  lower coefficients and **0.762** at 0.3 — the strong mass term interferes
  with avoidance exactly as the expected-degeneracy note anticipated. One
  seed, one observation; but it is the reason 0.03 ≤ 0.1 ≤ 0.3 was a ladder
  worth pre-registering.

## What this does and does not show

**Does:** leaving the move vocabulary is no longer free — `-log P(move)`
holds emission at ~1.0 for every seed and coefficient, at zero apparent cost
to the seeds that were going to avoid anyway (s1: 0.038, s2: 0.000 — among
the strongest avoiders in the whole programme, *while answering the
question*, which no wrecked E16 "functional" organism could claim).

**Does not:** fix the avoidance lottery. Seeds 3 and 4 emit perfectly and
wander anyway, at every coefficient — the per-seed outcome is nearly
coefficient-independent in the passing range, so what remains is the
training-draw variance E15 measured, now cleanly separated from the
emission defect that used to mask it.

## Threats

- **The avoidance bar was vacuous** (no non-wrecked control seeds). The
  honest external context: E16's healthy ORG-As averaged ≈ 0.31 with a wide
  spread; 0.03's 0.429 over disease seeds is consistent with that spread,
  not obviously worse. A future sweep wanting a live bar should add one
  healthy seed (e.g. seed 0) to the grid.
- **n = 4 seeds, chosen where the disease was** — deliberately, per
  pre-commitment (1); this licenses "emission restored on the wrecked
  seeds", not population statements.
- Entropy columns show no fixed-bonus-style collapse pattern attributable to
  the mass term (three exact-zero finals scattered across arms including
  coef 0.03 s3 — the seeds that wander, not the ones that avoid).

## Next

1. **The full corrected loading-map re-run at 0.03** — config already wired
   (corrected criteria, fixed-frame probe, per-kind volumes, A′ soft
   targets, extended battery, adapter persistence).
2. Run it **seed-parallel**: every (seed, kind) cell begins with
   `set_all_seeds(seed)`, so cells are RNG-independent by construction and
   process-level parallelism reproduces the sequential run exactly — this
   run just demonstrated per-cell reproduction across machines.
