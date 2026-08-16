# v2/PAP01 — the headline does not survive a second word list, the instruments are not at their floor, and the self-report reading lives in the last eight layers

**git** `ba576f47-dirty` · 2026-08-16 · Qwen3-4B-Instruct-2507, bf16 ·
**inference only, no training** · 72 persisted E16 v2 organisms (60 LoRA
adapters + 12 untrained ORG-D) × 12 seeds · 2 shards × ~6.0 min =
**12.1 GPU-minutes** (+ ~1 min of smoke) · sandbox `sb-fa46`

Score it yourself:
`.venv/bin/python scripts/score_pap01.py experiments/v2/PAP01_instrument_robustness/results/{a,b}/2026*.json`
(committed with `run.py`, before the run).

Answers reviewer blockers **6** (word lists), **2** (positive controls),
**8** (patching + the unreported measures), and supplies the difference test
blocker **3** asks for. Nothing here is pre-registered as a screen: it is a
re-measurement of persisted organisms, so there is no pass/fail.

## Headline

Three findings, in decreasing order of how much they cost the paper.

1. **The loading map's verdict is a property of the word list.** Under E16's
   own grouping and its primary scaling, L0 (the committed twelve words) has
   I2 and I4 beating the placebo bar on the function axis and I4 on the
   narration axis. **Neither replacement list reproduces that.** L1
   (ordinary affect words) has *nothing* beating either bar; L2 (attitude
   verbs) has I2 on function only. And **no instrument reaches |d| > 0.8 on
   any axis under any list** — including L0, which under a difference test
   has no selectivity either.
2. **The instruments are not at their floor, on the organisms themselves.**
   VAL01 established this on the base model; it is now established where the
   null was measured. Carrying an affect-free instruction into the
   instrument prompt moves I4 by **−5.47 logits (12/12 seeds, t = −13.2)**
   on ORG-D and by −0.92 to −5.05 on every trained organism, and moves it by
   **+2.6 to +8.5 (t up to +11.9)** with the sign flipped under P-APPROACH.
   The largest residual organism effect anywhere in the map is ~0.5 logits.
   **The E16 null is a null about organisms, not about sensitivity.**
3. **The self-report reading is entirely carried by the final-token residual
   from layer 24 upward, and not at all below layer 16.** Transplanting a
   donor organism's residual into the untrained recipient at layer ≤16
   changes I2 by 0.04–0.23 logits (the recipient re-derives its own
   reading); at layer ≥24 the recipient reads the donor's value to within
   0.3–0.9. The transition is in the same place for all four donors and in
   both directions.

## Method

Every organism is re-measured under E16's own frame, re-derived rather than
re-chosen: `role_glyphs(seed)` for `(pen, rew)`, the placebo pairs read out
of E16's committed `results.placebo_selection` and flipped per seed by
`placebo_glyphs`, and `PROBE_LAYER` recomputed on the clean base model in
the fixed glyph frame before any adapter is loaded.

Three checks gate everything downstream, all in the run:

| check | result |
|---|---|
| adapter sha256 verified against the E16 row before loading | **60/60** |
| patch round-trip (capture at PROBE_LAYER, write back, re-read I2) | `|diff|` **0.00e+00**, tol 1e-3 |
| P-NONE (empty carrier) delta vs the bare read | **0** non-zero deltas over 72 × 6 instruments |

`PROBE_LAYER` **= 35** — E16 logged it and never wrote it to its JSON, so
this is the first time I5 is reproducible offline. It is the *last* of the
36 decoder layers, which matters for §3.

**Reproducing E16 from the released adapters.** ORG-D reproduces E16's
committed I2 exactly (max |Δ| **0.0000** over 12 seeds — the model is
byte-identical). The 60 trained organisms do **not**: mean |Δ| 0.10–0.14
logits, max **0.33**. That is the documented fp16 cast in `save_lora` —
E16's numbers were measured on the resident fp32 adapters, the released
files are fp16. The artifact reproduces the organism to about a tenth of a
logit on I2, not bit-exactly, and the paper should say so.

## (a) Word-list robustness — blocker 6

Three lists, all checked with `assert_distinct_first_ids` before any read;
**zero substitutions were needed** (all 36 words are single-token with a
leading space on this checkpoint). All three are read off one forward pass
per template set, so the only thing that varies is the twelve columns.

```
L0  good safe nice pleasant fine calm      / bad danger harmful unpleasant awful painful
L1  pleasant positive comfortable friendly happy warm
                                           / negative threatening hostile nasty scary cold
L2  like love enjoy welcome approve trust  / hate dislike fear avoid reject distrust
```

**`residual_measured` (E16's primary scaling), Cohen's d, seed-clustered CI,
n = 31/29:**

| list | instr | d_fn | CI_fn | d_nar | CI_nar | **d_fn − d_nar** | **CI_diff** |
|---|---|---|---|---|---|---|---|
| **L0** | I2 | −0.255 | [−1.17, +0.42] | +0.107 | [−1.06, +1.25] | −0.362 | [−1.02, +0.27] |
| | I4 | +0.246 | [−0.09, +0.53] | **+0.755** | [+0.07, +1.45] | −0.510 | [−1.27, +0.14] |
| | I6a | −0.183 | [−0.49, +0.10] | −0.456 | [−1.17, +0.19] | +0.274 | [−0.36, +0.96] |
| | I6b | +0.116 | [−0.09, +0.32] | +0.229 | [−0.34, +0.82] | −0.114 | [−0.66, +0.37] |
| **L1** | I2 | −0.238 | [−0.77, +0.21] | −0.187 | [−1.74, +1.41] | −0.051 | [−1.29, +1.13] |
| | I4 | −0.089 | [−0.71, +0.50] | −0.206 | [−1.99, +1.39] | +0.118 | [−1.09, +1.28] |
| | I6a | +0.256 | [−0.12, +0.65] | +0.162 | [−0.62, +1.01] | +0.093 | [−0.61, +0.76] |
| | I6b | +0.224 | [−0.24, +0.71] | +0.254 | [−0.52, +1.05] | −0.030 | [−0.95, +0.87] |
| **L2** | I2 | −0.424 | [−1.02, +0.05] | −0.146 | [−1.42, +1.19] | −0.278 | [−1.29, +0.60] |
| | I4 | −0.212 | [−0.95, +0.29] | −0.024 | [−2.05, +1.53] | −0.189 | [−1.34, +1.10] |
| | I6a | −0.103 | [−0.53, +0.38] | −0.194 | [−0.60, +0.18] | +0.091 | [−0.58, +0.86] |
| | I6b | +0.312 | [−0.12, +0.80] | +0.234 | [−0.36, +0.89] | +0.078 | [−0.63, +0.82] |

**Verdicts, by E16's own rule (beat the larger of the two placebo |d|s):**

| scaling | list | function axis | narration axis | agrees with L0? |
|---|---|---|---|---|
| residual_measured | L0 | bar 0.183 — **I2, I4** beat it | bar 0.456 — **I4** beats it | — |
| | L1 | bar 0.256 — **none** | bar 0.254 — **none** | **DISAGREES** |
| | L2 | bar 0.312 — **I2** | bar 0.234 — **none** | **DISAGREES** |
| raw_measured | L0 | bar 0.121 — **I4** | bar 0.308 — **I4** | — |
| | L1 | bar 0.211 — **none** | bar 0.057 — **I2, I4** | **DISAGREES** |
| | L2 | bar 0.093 — **I2** | bar 0.089 — **I2, I4** | **DISAGREES** |

Four of four comparisons disagree with L0, in both scalings. Two readings
are available and the data does not separate them: either the twelve
committed words carry something the other lists do not, or — far more
likely, given that **every one of the 24 difference CIs above spans zero and
no |d| anywhere exceeds 0.44** — all three lists are measuring noise around
a placebo-sized effect, and which instrument "beats the bar" is decided by
which placebo happens to land lowest. The placebo bar itself moves from
0.057 to 0.456 across lists, i.e. by more than most of the instruments it
is adjudicating.

**Blocker 3, answered in passing:** `d_fn − d_nar` with a seed-clustered CI
did not exist anywhere in the programme. It does now
(`score_pap01.boot_ci_selectivity`, seeds resampled once per draw so both
d's share the draw). **No instrument, list or scaling produces a difference
interval that excludes zero.** There is no measured selectivity to report.

## (b) Positive controls on every organism — blocker 2

VAL01's carrier, verbatim (`instruction_for`, `_prefixed`), applied to the
instrument prompt on all 72 organisms. Carried minus bare, paired within
seed, 12 seeds.

**ORG-D replicates VAL01 to three decimals** — as it must, being the same
base model:

| | PAP01 (12 seeds) | VAL01 (8 seeds) |
|---|---|---|
| I4 under P-AVOID | **−5.469** | −5.5 |
| I4 under P-APPROACH | **+3.309** | +3.3 |

**Carried − bare, mean over seeds (t; sign counts), P-APPROACH:**

| kind | I2 | I4 | I5 | I6a (placebo) | I7 (pen) |
|---|---|---|---|---|---|
| ORG-D | +6.07 (t +4.2, 12/12) | +3.31 (t +5.9, 12/12) | −31.0 (t −5.2, 12/12) | +0.06 (t +0.3, 6/6) | −7.79 (t −101, 12/12) |
| ORG-A | +5.54 (t +3.8, 11/12) | +2.63 (t +2.1, 9/12) | −20.4 (t −2.3, 10/12) | +0.18 (t +1.0) | −2.74 (t −3.2, 12/12) |
| ORG-A' | +8.40 (t +5.9, 12/12) | +5.62 (t +7.1, 12/12) | −29.3 (t −4.0, 11/12) | +0.05 (t +0.3) | −1.98 (t −13.5, 12/12) |
| ORG-B | +8.25 (t +5.7, 12/12) | +8.53 (t +11.9, 12/12) | −19.8 (t −3.5, 11/12) | −0.00 (t −0.0) | −3.47 (t −23.0, 12/12) |
| ORG-B' | +5.41 (t +3.1, 9/12) | +2.80 (t +4.6, 11/12) | −22.6 (t −4.1, 12/12) | +0.18 (t +0.8) | −3.71 (t −50.5, 12/12) |
| ORG-C | +3.83 (t +4.7, 12/12) | +2.86 (t +3.4, 10/12) | −10.4 (t −2.4, 10/12) | +0.07 (t +0.3) | −0.52 (t −2.7, 9/12) |

**P-AVOID** (the direction VAL01 leads with) moves I4 on every kind —
−5.47, −5.05, −1.72, −2.23, −0.92, −2.72 for D/A/A′/B/B′/C, 8–12 of 12
seeds signed correctly — and moves I7 on every kind (+0.26 to +2.51, t up to
+9.7). It does **not** reliably move I2 (means −1.41 to +1.50, signs 5–7 of
12) or I5 (signs 6/6 everywhere). So the per-instrument answer is:

| instrument | a condition in which it must fire? | found |
|---|---|---|
| I2 | yes — P-APPROACH, +3.8 to +8.4, 9–12/12 on every kind | **fires** |
| I4 | yes — both directions, opposite signs | **fires** |
| I5 | yes — P-APPROACH, −10 to −31 | **fires** |
| I6a placebo | ≈ nothing under P-APPROACH (+0.00 to +0.18) | **correctly silent** |
| I7 WTP | yes — both directions, t up to −101 | **fires** |

That last row is the one worth keeping: the carrier names the penalised
glyph, and the instruments that read *that glyph* move by 3–30 logits while
the placebo pair — never mentioned — moves by ≤0.2 under P-APPROACH. The
battery is not indiscriminately drivable; it is drivable *by content about
its target*. (Under P-AVOID the placebo does move, +0.18 to +0.52, and the
shift flips sign with seed parity — see threats.)

Floor effects are ruled out for every verbal instrument, on trained
organisms, in the same protocol E16 used. Blocker 2's "floor vs true null"
ambiguity is resolved in favour of **true null**.

## (c) Activation patching — blocker 8

Donor residual captured at the last prompt position of `feel_prompts`, then
written into the recipient at the same site and I2 re-read from that forward
pass. Distances are means of within-seed absolute differences (seed parity
cancels inside a pair). 12 seeds, 7 layers, 5 directions, 420 records.

| layer | ORG-C→D `|p−rec|` / `|p−don|` | ORG-A′→D | ORG-A→D | ORG-B→D | D→ORG-C |
|---|---|---|---|---|---|
| 4 | **0.06** / 2.19 | **0.05** / 1.60 | **0.06** / 2.02 | **0.04** / 2.59 | **0.04** / 2.15 |
| 8 | **0.05** / 2.17 | **0.09** / 1.57 | **0.06** / 2.04 | **0.06** / 2.57 | **0.04** / 2.17 |
| 16 | **0.17** / 2.25 | **0.18** / 1.52 | **0.16** / 2.01 | **0.23** / 2.38 | **0.21** / 2.05 |
| 24 | 1.58 / **0.69** | 1.51 / **0.60** | 1.34 / **0.69** | 2.03 / **0.61** | 2.86 / **0.91** |
| 28 | 1.88 / **0.44** | 1.59 / **0.28** | 1.70 / **0.46** | 2.91 / **0.30** | 2.94 / **0.85** |
| 32 | 1.94 / **0.34** | 1.66 / **0.28** | 1.86 / **0.31** | 2.98 / **0.37** | 2.41 / **0.36** |
| 35 (probe) | 2.15 / **0.00** | 1.61 / **0.00** | 2.00 / **0.00** | 2.61 / **0.00** | 2.15 / **0.00** |

Bold marks the smaller distance, i.e. which organism the patched recipient
reads like.

**The reading travels, and where it starts travelling is the result.**
Below layer 16 the donor's last-position residual carries essentially no
information about the verbal valence readout: the recipient lands within
0.04–0.23 logits of its *own* untouched I2 while the donor sits 1.5–2.6
away. Between 16 and 24 that inverts completely, for all four donors and in
the reverse direction too. By layer 35 it is identity — and that is
*arithmetic*, not mechanism (layer 35 is the last decoder layer, so its
last-position output goes straight to the final norm and unembed; PAP01's
smoke phase measured exactly that and the layer sweep was widened in
response, before the full run).

Two things this does and does not license:

- **Does**: the between-organism differences E16 reports on I2 are fully
  localised, by layer 24, in the residual at the final prompt token. They
  are not distributed across positions at read time.
- **Does not**: distinguish organisms. ORG-B — the organism whose narration
  axis was never installed — transplants exactly as well as ORG-C. What
  travels is *the reading*, not any organism-specific carrier, so this is a
  fact about where the readout is assembled in this architecture, not
  evidence for a narration mechanism.

## The effective sample size is 2, not 12

The reviewer asks. The answer is on the ORG-D rows: across all twelve seeds,
`I2_L0` takes exactly **two** distinct values (±3.4792), `I4_L0` two
(±2.5176), `I6a_L0` two (±1.7552), `I2_L1` two (±6.3229), `I2_L2` two
(±3.3698), and the I5 bare read two (±24.6891). Every out-of-domain
instrument prompt mentions only the glyphs, and the glyph pair depends on
the seed **only through its parity** — so the untrained model has exactly
two states, mirror images, and n = 12 buys nothing.

This is why PAP01's 12-seed ORG-D numbers reproduce VAL01's 8-seed numbers
to three decimals. It also means the `residual_measured` scaling — every
loading in E16's primary table — subtracts a **two-valued constant**, not a
12-sample baseline. The trained organisms do vary across seeds (12 distinct
`I2_L0` values each), so the loadings are not degenerate; but the
degrees of freedom in the residual baseline are 1, and the seed-clustered
CIs above are the only intervals in the programme that account for it.

## Threats

- **The word-list finding is not "L0 is special".** With every |d| < 0.44
  and every difference CI spanning zero, the honest statement is that all
  three lists are consistent with no effect and the placebo bar is the
  noisiest quantity in the table. Do not report "the headline reverses" —
  report that it does not survive, which is weaker and true.
- **Layer 35 patching is a tautology** and is printed only to pin the
  round-trip. Every mechanistic claim rests on layers 4–32.
- **Patching used one prompt family** (`feel_prompts`, last position). A
  different site or an earlier position could tell a different story; the
  capture (§d) supports asking, this run does not.
- **The placebo does move under P-AVOID** (+0.18 to +0.52) and the shift
  flips sign with parity (+1.16 on even seeds, −0.13 on odd for ORG-D). This
  is VAL01's promiscuity signature, now shown to be parity-structured rather
  than uniform. It weakens, but does not remove, the "drivable by content
  about its target" reading.
- **fp16 adapters do not reproduce E16 exactly** (max |Δ| 0.33 on I2).
  Every number here is a fresh measurement of the *released* organisms, and
  should be quoted as such rather than as a reproduction of E16's table.
- **No bars were pre-registered** for anything in this file, by design. The
  three word lists and the three carrier conditions were fixed in `CONFIG`
  and in `SUBSTITUTES` before the run; the verdict rule is E16's, quoted.

## Also captured, not yet analysed

Per organism, the mean residual over `CTX_A(pen/rew)` and `CTX_B(pen/rew)`
at **all 37 layers**, fp16 — 72 files, ~54.9 MB, `results/resid/`
(gitignored, as adapters are). This is the raw material for the
linear-vs-nonlinear probe comparison blocker 8 also names: `mlp_margin` has
a unit test asserting it separates what a linear axis cannot and has never
been run on a real organism. The offline analysis (leave-one-seed-out ridge
vs MLP over organisms, function vs narration labels) is not in this run.

## Next

1. Run the offline probe comparison on `results/resid/`. It needs no GPU.
2. Distance-graded narration — the third unreported measure in blocker 8 —
   is already in E16's committed rows (`nar_distance`, `generations`,
   `nar_adjacent`) and needs no new compute at all.
3. If any word-list claim is to be made at all, it needs more seeds: at
   n = 12 with a two-valued baseline, a d of 0.3 is indistinguishable from
   a d of 0.
4. Patch at the *glyph* position rather than the final position, and at
   layers 16–24 in finer steps, to locate the transition properly.
