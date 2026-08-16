# Compute cost — every run to date, and the queue

**2026-08-14.** Derived from 43 committed run manifests (`started_at`/`finished_at`,
`results.elapsed_minutes`) across `experiments/**/*.json`. Three runs without manifests
(E2a, E3, E4) are estimated from their RESULTS.md headlines and marked `est`.

Hardware: NVIDIA RTX PRO 6000 Blackwell, 102 GB, 20 CPU (HANDOFF.md §3), reached over
HTTP from marimo sandboxes.

Interactive version with the full per-run tables:
https://claude.ai/code/artifact/b3c1f2bc-c709-474e-9436-83fb850715db

---

## Headline

| basis | GPU-h | low | mid | high |
|---|---:|---:|---:|---:|
| Job time (sequential-equivalent) | 23.70 | $16.59 | $42.66 | $82.95 |
| GPU-busy (actual silicon, per box) | 16.94 | $11.86 | $30.49 | $59.29 |
| **Rented (box-hours held — billable)** | **24.71** | **$17.30** | **$44.48** | **$86.48** |

Rates: low $0.70/GPU-h (spot / owned-card marginal), mid $1.80 (on-demand secure cloud),
high $3.50 (managed platform or H100 substitution).

⚠️ **Cash actually spent may be $0.** The runs went through molab.run sandboxes. If those
are free-tier, every figure here is *replacement cost* — what reproducing or scaling this
would take — not outlay.

### Why the three numbers differ

- **Job time** sums every run's own wall clock. E16 v2's six shards each ran ~50 min, so
  it contributes 301 min even though they overlapped.
- **GPU-busy** is the union of busy intervals per box. Those six shards time-slice one
  card, so they contribute ~52 min.
- **Rented** is box-hours held from first run to last, idle included. Half of the two long
  sessions was idle GPU.

---

## Per act

| act | runs | min | GPU-h | low | mid | high |
|---|---:|---:|---:|---:|---:|---:|
| Act 1 — detector hunt (07-30) | 10 | 43.6 | 0.73 | $0.51 | $1.31 | $2.54 |
| Act 2 — organism yield (07-30) | 3 | 51.8 | 0.86 | $0.60 | $1.55 | $3.02 |
| Act 3 v1 — build & first maps | 9 | 243.1 | 4.05 | $2.84 | $7.29 | $14.18 |
| Act 3 post-audit (08-11→14) | 4 | 613.4 | 10.22 | $7.16 | $18.40 | $35.78 |
| v2 track (08-13→14) | 9 | 470.1 | 7.84 | $5.48 | $14.10 | $27.42 |
| **total (job time)** | **35** | **1422.0** | **23.70** | **$16.59** | **$42.66** | **$82.95** |

Largest single line items: E16 v2 final (301.1 min job / 52 min wall, 6 shards),
E16 v2 attempt 1 (197.9 min, superseded), SCL01 32B (108.1), E17 (96.9), SCL01 14B (95.3).

---

## Session occupancy

| campaign | held | busy | util | runs |
|---|---:|---:|---:|---:|
| 07-30 14:43 → 22:05 | 7.36 h | 3.67 h | 50% | 19 |
| 07-31 08:52 → 10:42 | 1.83 h | 1.82 h | 99% | 2 |
| 08-11 06:11 → 07:48 | 1.62 h | 1.62 h | 100% | 1 |
| 08-13 20:48 → 08-14 04:02 (box M) | 7.23 h | 3.80 h | 53% | 12 |
| 08-13 21:28 → 08-14 04:08 (box S) | 6.67 h | 6.03 h | 90% | 8 |
| **total** | **24.71 h** | **16.94 h** | **69%** | **42** |

The idle half of the two long sessions is a bigger recoverable loss than every superseded
run combined.

## Redo tax

| measure | GPU-h | share | includes |
|---|---:|---:|---|
| narrow | 2.85 | 12% | E1b v1, E1c, E4, E12 ×2, E13 v1+v2, E14, E16 v1 |
| broad | 6.15 | 26% | + E16 v2 attempt 1 |

12–26% is low for this much methodological churn, and that is the point: the
audit-and-repeat discipline in REVIEW.md is affordable *because* a run costs minutes.

---

## Two facts that should drive every future estimate

### 1. Parameters barely move the cost

A 53× parameter increase bought **1.53×** runtime for identical work. The workload is 800
sequential optimizer steps at batch 8 on ~70-token prompts — latency- and
kernel-launch-bound, not FLOP-bound.

| size | minutes | min/organism | vs 0.6B |
|---|---:|---:|---:|
| 0.6B | 70.7 | 1.96 | 1.00× |
| 1.7B | 70.2 | 1.95 | 0.99× |
| 4B | 79.9 | 2.22 | 1.13× |
| 8B ⚠️ | 36.5 | 1.01 | 0.52× |
| 14B | 95.3 | 2.65 | 1.35× |
| 32B | 108.1 | 3.00 | 1.53× |

⚠️ The 8B row is **not** a scaling data point — SCL01's scorer marks its rows invalid
(non-monotone in size, template/recipe interaction) and the short runtime is consistent
with degenerate training. Excluded from the trend.

**Consequence: moving the workhorse from 4B to 14B costs ~19% more, not 3.5×.**
HANDOFF.md's open question ("14B as the program's workhorse size?") is cheap to answer yes.

### 2. Parallelism is nearly free

E16 v2: 301 min of work in 52 min wall across 6 shards on one card — **5.79× at 96%
parallel efficiency**. Near-linear scaling means a single-stream run leaves ~83% of the
card idle. `scripts/e16_parallel.py` already does this and is bit-identical to sequential
via per-cell `set_all_seeds`. Shard count could likely go past 6 before saturating 102 GB.

**Buy seeds and shards, not a bigger GPU or a smaller model.** A 14B, 12-seed, 6-way
sharded map is roughly a one-hour wall-clock job on this hardware.

---

## FLOPs (hardware-independent, ±2× band)

| unit | tokens | FLOPs |
|---|---:|---:|
| one ORG-A (RL, 4B) | ~450 k | ~7e15 |
| one ORG-A′ (SFT, 4B) | ~260 k | ~4e15 |
| one 6-kind seed (4B) | ~1.6 M | ~2.6e16 |
| E16 v2 (72 organisms) | ~19 M | ~3e17 |
| SCL01 ladder (6 sizes) | ~58 M | ~2.2e18 |
| **whole program** | — | **~4e18** (3–6 EFLOP) |

Cross-check: 4e18 over 16.94 GPU-busy hours = ~70 TFLOP/s sustained against a
~200 TFLOP/s bf16-class card — ~35% utilization, far lower single-stream. Same story as
fact 2 in different units.

For scale: GPT-3 training was ~3e23 FLOPs. This program is ~1e-5 of that. **The science
here has never been compute-limited.**

---

## Forward book

From HANDOFF.md Addendum 5 queue items 6–11, plus REVIEW.md R15 and E16 v2's open coef
question. GPU-hours:

| # | item | low | mid | high | basis |
|---|---|---:|---:|---:|---|
| 6 | ENV02 rebuild (gentler recipe) | 0.3 | 0.8 | 2.0 | ENV02 was 8.4 min; 2–4 attempts |
| 7 | NAR01 factorial (content × contingency) | 2.0 | 6.0 | 15.0 | E17-class (72 org, 1.6 h) × 2–3 |
| 8 | VAL01 follow-up (trained arm) | 1.0 | 3.0 | 7.0 | VAL01 needed no training; this does |
| ★ | 14B workhorse re-map, E16-class + battery | 3.0 | 6.0 | 12.0 | E16 v2 job time × 1.19 size factor |
| ★ | Missing instruments (cost-paying, GARP, refusal, +controls) | 0.5 | 2.0 | 5.0 | mostly inference on persisted adapters |
| ★ | ORG-B three vocabulary intensities | 0.5 | 2.0 | 5.0 | SFT-only arms, E13-class each |
| ★ | coef 0.05 / per-seed acquisition-gate sweeps | 0.5 | 1.5 | 4.0 | E18 was 0.29 h per sweep |
| 9–11 | lit review, C11 fix, post draft | 0 | 0 | 0 | no GPU |
| | subtotal | 7.8 | 21.3 | 50.0 | |
| | × redo factor (1.10 / 1.30 / 1.80) | 8.6 | 27.7 | 90.0 | from the measured 12–26% |
| | **future $** | **$6** | **$50** | **$315** | |

NAR01 is the only item that could plausibly blow past its estimate — it is designed and
coded from scratch, and this program's from-scratch experiments have historically needed
2–3 runs.

### Program total

| scenario | past (rented) | future | total GPU-h | total $ |
|---|---:|---:|---:|---:|
| low | 24.7 | 8.6 | 33.3 | $23 |
| **mid** | **24.7** | **27.7** | **52.4** | **$94** |
| high | 24.7 | 90.0 | 114.7 | $401 |

**Past and planned, the whole program lands between $23 and $401 of GPU. Realistically
~$94 and 52 GPU-hours — about two days of one rented card.**

---

## The cost that isn't GPU

HANDOFF.md said it after E2a: *"the binding constraint is engineering and analysis time,
not the GPU."* Fourteen days and ~148,000 words of committed code and prose later, that
still holds. If the authoring was agent-driven, the inference bill plausibly **exceeds**
the GPU bill.

Rough band — **estimated from committed text volume, not from usage records**:

| component | estimate | rate | cost |
|---|---:|---:|---:|
| surviving output (committed .md + .py) | ~200 k tok | — | — |
| total output incl. discarded drafts (3–10×) | 0.6–2 M tok | $25/M | $15–50 |
| fresh input | 10–40 M tok | $5/M | $50–200 |
| cache-read input | 40–150 M tok | $0.50/M | $20–75 |
| **order of magnitude** | | Opus-class | **$85–325** |

Treat as an order of magnitude, not an estimate. The point survives the imprecision:
**the GPU is the cheap half of this program.**

---

## What follows for planning

1. **Stop treating GPU cost as a design constraint.** At $1.80/h a 72-organism map is $9.
   Cut instruments and seeds for scientific reasons only — HANDOFF.md §7 rule 1 already
   says this; the ledger confirms it is affordable.
2. **Promote 14B to the workhorse.** First size where the RL recipe is reliable
   (ORG-A 6/6) and pre-commitment (1) would pass; costs ~19% more than 4B.
3. **Shard every map by default.** 96% efficiency at 6 workers; try 8–12.
4. **Close the idle gap.** Queue the next run before the current one finishes — the
   auto-chain markers used on 08-13 should be the default, not the exception.
5. **Budget the queue at ~28 GPU-h / $50**, with $315 as the ceiling.

---

*Rates are market estimates for RTX PRO 6000 Blackwell class hardware as of 2026-08 and
vary by provider, region, and commitment. Model pricing per Claude Opus 5 list rates
($5/$25 per MTok; cache reads ~0.1×). FLOP figures carry a ±2× band.*
