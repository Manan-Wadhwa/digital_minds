# v2/NAR01 — fitting the remark distribution makes the suffix collapse worse, and the pools are why

**git** `c920e7ca` (clean) · 2026-08-14 · Qwen3-4B + LoRA r=16 · 8 seeds × 4 arms
× 3 kinds = **96 organisms** · **66.9 min GPU** (32.9 + 34.0 on two boxes
concurrently; **34 min wall**) · cuda:0 × 2 (RTX PRO 6000 Blackwell)

Score it yourself: `python3 scripts/score_nar01.py
experiments/v2/NAR01_narration_recipe/results/20260814T113107Z_e9c4426a6aa4.json`
— stdlib only, committed in `f67f17cc`, two commits before any number existed.

---

## Headline

**The fix that repaired ORG-A′ does not repair ORG-B. Applied to the remark it
is not merely inert — it doubles the suffix collapse, and the cause is that
`AVERSIVE` has six members and `FILLER` has four.**

ORG-A′ and ORG-B fail the same way (E17: fitting a *sample* from a flat pool,
71.5% of the remark gradient is noise). ORG-A′'s repair was to fit the
**distribution** instead (`sft.soft_move_target`). Nobody had applied it to the
remark. This run did, by emitting every pool member per state, and it went
backwards:

| arm | mean contingency | > 0.5 bar | ORG-B invariant | ORG-B′ invariant | suffix collapse | distinct grids |
|---|---:|---:|---:|---:|---:|---:|
| control | +0.112 | 0/8 | 7/8 | 7/8 | 4/8 | 1536 |
| **pool1** | **+0.429** | **3/8** | **7/8** | **8/8** | **1/8** | 1536 |
| enum | +0.106 | 1/8 | 5/8 | **4/8** | **6/8** | ~292 |
| enum_bal | +0.314 | 1/8 | 7/8 | 7/8 | 2/8 | 307 |

*Suffix collapse = seeds with narration presence ≥ 0.90 on **both** adjacent and
non-adjacent states — `organisms.py`'s "has not learned to talk about the tile;
it has learned a suffix".*

Paired over 8 seeds: `enum − control = −0.007`, `enum − pool1 = −0.323`.
Pre-commitment (2) asked `enum` to beat control and be non-inferior to pool1.
**It did neither, and pre-commitment (1) rejects it independently** — ORG-B′,
whose only job is to sit still, holds policy on 4/8 seeds against pool1's 8/8.

**E17's pool1 stands, at exactly the 3/8 it scored in E17, on eight fresh
seeds.** A valid ORG-B still does not exist.

---

## Why — unequal pools silently reweight the classes

`enum` has the **highest** narration presence of any arm (0.965 adjacent) and
the **lowest** contingency. It says an aversive remark on **86% of states where
the penalised tile is not even adjacent**. That is not weak learning; it is the
marginal, learned and then saturated by greedy decoding — E16's seeds 4/9/11
failure mode, which pool1 exists to prevent.

The cause is arithmetic. Enumeration emits one example per pool member, and the
two pools are **not the same size**: `AVERSIVE` has 6 entries, `FILLER` has 4.
So "fit the conditional distribution" silently multiplies the adjacent class by
6 and the non-adjacent class by 4, and the aversive share of the corpus is no
longer the adjacency rate:

| arm | corpus adjacency | predicted aversive-example share | observed non-adjacent presence |
|---|---:|---:|---:|
| control | 0.641 | 0.641 | 0.663 |
| enum | 0.620 | **0.710** | **0.859** |
| enum_bal | 0.502 | 0.602 | 0.545 |
| pool1 | 0.633 | 0.633 | **0.476** |

For every arm in which the model falls back on the marginal, the predicted share
tracks the observed non-adjacent presence and **orders the arms correctly**:
enum 0.710 > control 0.641 > enum_bal 0.602. Enumeration raised the free-lunch
marginal from 62% to 71%, and greedy decoding did the rest.

**pool1 is the one arm that breaks the pattern** (predicted 0.633, observed
0.476) — which is E17's finding restated: it is the only arm where the
within-pool term is zero, so the conditional is learnable and the model does not
have to settle for the marginal.

This retires the whole family, not just this arm: **any scheme that hands the
remark its conditional distribution by enumeration will reweight the classes
whenever the pools differ in size.** The soft-move target never hit this because
its target is a 4-way distribution over one categorical position, with no
"pool" whose cardinality can differ by condition.

It also names the cheap fix: enumerate with per-member weight `1/|pool|` so each
state contributes equal mass regardless of pool size — or simply make the pools
equal length. Neither was tested here.

---

## Findings

1. **Enumeration is not inert, it is harmful.** `enum − control = −0.007` looks
   like nothing, but the presence numbers show why that is a coincidence of two
   opposed effects: enumeration removes within-pool noise (helping) while
   raising the aversive marginal from 0.62 to 0.71 (hurting more). Suffix
   collapse goes 4/8 → 6/8.

2. **Enumeration moves the policy.** ORG-B′ is the clean readout — no affect, its
   only job is to sit still: **4/8 invariant under `enum` vs 8/8 under `pool1`**
   and 7/8 under control. Seed 1 drifts in *every* arm (ratio 1.18–1.22), so it
   is a drifty seed rather than an arm effect; `enum` adds seeds 2 and 6 on top.
   5.3× more gradient on remark tokens at matched example count is a sufficient
   explanation. Under pre-commitment (1) this rejects the arm on its own.

3. **Balancing adjacency is worth more than E17 credited it.**
   `enum_bal − enum = +0.208` at near-identical state counts (307 vs ~292) and
   identical example counts, and it cuts suffix collapse 6/8 → 2/8. E17
   introduced balancing as a minor adjunct to pool1 ("barely moves the signal
   share, 28.5% → 30%") and could not isolate it; here it is the only difference
   between two arms. Consistent with the mechanism above: balancing pulls the
   predicted aversive share from 0.710 back to 0.602.

4. **The state-diversity confound did not decide this.** Pre-commitment (3)
   named it the first suspect if `enum` lost. It is not sufficient: `enum` and
   `enum_bal` differ by +0.208 at nearly the same grid count (~292 vs 307),
   while `enum` and `control` differ by −0.007 across a 5.3× grid gap. Grid
   count does not order these results; corpus composition does.

5. **The canaries are clean.** ORG-D is bit-identical across all four arms
   (`[0.941, 1.132, 0.983, 0.926, …]`), which is what per-cell `set_all_seeds`
   promises and what makes the two-box shard merge legitimate. ORG-B′ mean
   contingency is +0.000 in every arm, so the contrast measures affect and not
   talkativeness.

6. **The strict detector is not the problem.** On the stored samples the
   paraphrase-tolerant `says_aversive_lexical` and the pre-registered strict
   `_says_aversive` disagree on **1 of 96 texts**. `enum`'s near-zero contingency
   is not a matcher artefact; the organism really does narrate on both classes.

---

## Threats

- **A first draft of this file claimed the wrong mechanism.** It read three
  generations from one seed, saw run-on text, and asserted enumeration causes
  sentence concatenation — a distribution-over-sentences argument. Counting all
  96 stored texts killed it (concatenation appears in 1/24 `enum` samples and
  4/24 `pool1` samples, the wrong direction), and the presence table gave the
  real answer. Recorded because the eyeball reading was persuasive and wrong,
  and because three pre-registered criteria in this programme have already
  passed on the wrong property.
- **`run.py` stores only 3 sample generations per row, not all 160.** The
  presence/contingency numbers are computed over the full audit set inside the
  run, so the findings stand, but **no text-level claim can be re-derived
  offline at scale.** Finding 6 rests on 96 texts, not 15,360. Future runs in
  this track should store the generations and the adjacency flags, as ENV02
  does.
- **The scorer's verdict line is wrong, and its own rows say so.**
  `score_nar01.py` treats pre-commitment (1) as all-or-nothing per arm, so one
  drifting seed prints `REJECTED` for every arm and the verdict collapses to
  `best policy-invariant arm = None`. Control drifts on 1/8 too — 1/8 is the
  baseline, not evidence. The defensible reading compares against control
  (pool1 1/8 ≈ control 1/8 → does not move policy; enum's ORG-B′ 4/8 → does),
  and that is the reading used above. **Typed, not fixed**, per the
  review-before-fix rule. The scorer's own last line — "Read the per-arm rows
  above, never this line alone" — is what saved it.
- **3/8 is not a majority; pool1 still fails E17's bar.** Nothing here builds a
  valid ORG-B. The best arm reproduces E17's number, it does not beat it.
- **n=8 and contingency is bimodal.** Several arms sit at exactly +0.000 on some
  seeds and +0.5–0.8 on others; means over 8 seeds of a bimodal quantity are
  weak summaries. The per-seed table in the scorer is the honest view.
- **The predicted-share table is arithmetic, not a fitted model.** It orders
  three arms correctly and pool1 deliberately departs from it. Four points is
  not a validation.
- **Adapters are split across two boxes.** The merge folded 32 of 64 ORG-B/B′
  adapters (shard_b's); shard_a's 32 remain on the first sandbox. Every
  `adapter_sha256` is in the committed JSON, so nothing scientific blocks, but
  the artifact set is incomplete until pulled.
- **`enum`'s example count is 1532–1536, not exactly 1536** — the corpus is cut
  at a whole-state boundary. At ≤0.3% this cannot carry the result.

---

## Next

1. **Equalise the pools, then re-test enumeration.** The mechanism predicts that
   `AVERSIVE[:4]` vs `FILLER[:4]` removes the reweighting entirely and returns
   `enum` to at least control. This is the direct test of the Headline claim and
   it is one arm, ~8 min of GPU. If enumeration still fails with equal pools,
   the reweighting explanation is wrong and should be retracted.
2. **Weighted enumeration.** Per-member weight `1/|pool|` gets the variance
   reduction without touching the class balance. Needs a `loss_weights`
   argument to `train_sft`; not currently supported.
3. **Store the generations.** Finding 6 and any future text-level claim need
   them. One line in `run.py`; ENV02 already does it.
4. **Isolate balancing directly.** Finding 3 makes `pool1_bal` vs `pool1` the
   cheapest open comparison, and E17 already ran `pool1_bal` — its committed
   data may answer this with no GPU at all.
5. **ORG-B may not be buildable by corpus construction.** Three experiments have
   now moved the number without clearing the bar (E17 +0.32; pool1 replicated
   here at +0.32; balancing +0.21). Every lever tried so far is a property of
   the *corpus*. The remaining lever that is not is **co-training**, which is
   what ORG-C has and ORG-B does not — and ORG-C is the only kind that reaches
   contingency reliably (E16 v2: 7/12 vs 0/12). NAR02 should be built around
   that, stating up front that only a co-trained ORG-B whose policy stays put
   counts as a build.
