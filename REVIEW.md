# REVIEW — an adversarial audit of this repository's claims

**Date:** 2026-08-14 · **Scope:** every experiment writeup (E1a–E17), README,
HANDOFF, `writing/post-draft.md`, `docs/*.html`, and the `src/calibration/`
package · **Method:** three independent deep audits (Act 1; Act 2/3 +
cross-document consistency; code/statistics), every headline-changing claim
then re-verified directly against the committed JSONs and git history.

**Reproduce it yourself:** `python3 scripts/rescore_review.py` (stdlib only, no
venv needed) re-derives every number this review claims — 108 checks, each
printed as *claimed vs recomputed*. Sections of that script are cited here as
`[S1]`–`[S16]`. In the repo's own convention: read the per-condition numbers,
not this review's verdict strings.

---

## 0. How to read this

This repository is unusually self-critical, and much of what it says about
itself is true. The bar for this review was therefore not "find mistakes" but:
find what the writeups **still** get wrong, where the self-criticism is
miscalibrated or asymmetrically applied, and what the committed data supports
that no document states.

The one-sentence summary: **the arithmetic is clean; the
provenance-of-inference is not.** Every quoted number traced to a committed
JSON reproduces exactly. What fails audit is *when the rules were written*,
*which withdrawals get honoured*, and *which of six equally pre-committed
analyses gets called "the" result*.

---

## 1. What holds up

Stated first, because it is substantial and it is the part a hostile reader
would be wrong to dismiss:

- **Every quoted number traced to committed JSON reproduces.** The E16
  headline table to 4 decimals `[S2]`, E17's paired effects `[S11]`, E15's
  lottery statistics `[S13]`, E13's ratios `[S15]`, E3/E4/E5/E6/E7's committed
  values `[S14]`. Nothing in this review found a fabricated or mistranscribed
  number.
- **E17 is a model experiment.** Clean provenance (`209cbc2`, no `-dirty`),
  pre-registered bar, honest FAIL on that bar, the paired difference reported
  instead of the flattering absolute level, and no bar-moving after the fact.
- **The placebo repair is real.** The E16 placebo-selection table reproduces
  (old pair's 4.844 pedestal included), counterbalancing is implemented as
  designed, and the placebos could have failed and did not — in the raw
  scalings.
- **Core numerics are correctly implemented**: `cohen_d`, the seed-clustered
  bootstrap's semantics, within-seed residualisation, `choice_gap`'s order
  control, the glyph token guard, Dr.GRPO/SFT/LoRA plumbing, RNG stream
  derivation, and the dual-ridge probe. The failure catalog in post §9–§10 is
  accurate and genuinely valuable.
- **The manipulation-check consolidation** (`manipulation.py`) and E17's
  contingency measure are real fixes to real defects the repo itself found.

---

## 2. Headline-changing findings

### R1. The trust-anchor provenance claim is false

> "Scored by a script we committed before the numbers existed, so we could not
> tune the criteria to them." — post-draft:80, repeated at post-draft:175,
> E16 RESULTS.md:6–8, HANDOFF.md:105

Git says otherwise `[S1]`: E16 finished **2026-07-31T10:42** (manifest
`finished_at`; `run.log` streams every reading from 09:27), and
`scripts/score_e16.py` was first committed **2026-08-03T13:59** (`81a94b9`) —
three days after the numbers existed, on a sandbox the author reconnected to
(`36f6df5`, "recover the completed 72-organism run off the ephemeral
sandbox", 14:20 the same day). E16 was additionally re-scored under revised
checks on 08-05 (`5bbe5ff`).

What **was** committed before the run — by 16 minutes — is `run.py` with its
docstring pre-commitments. That is a genuinely meaningful pre-registration and
it is the thing the sentence should say. The repeated claim, on the one
artifact readers are told anchors their trust, is the exact class of
provenance error the repo polices elsewhere.

### R2. The central pass rule was invented at writing time, is mis-stated, and does not generate its own table

> "The design fixed a pass rule in advance: an instrument counts as function
> selective if the confidence interval on its function loading excludes its
> narration loading." — post-draft:82

Three independent problems `[S1, S2]`:

1. **Not fixed in advance.** That rule appears in no committed code. It first
   appears in writing commit `667e411` ("score the results against confidence
   intervals"), 2026-08-05 — five days after the data. The scoring rule that
   *was* committed (score_e16.py, itself post-hoc, see R1) is a different
   rule: point thresholds |d| > 0.8 / |d| < 0.5.
2. **Does not generate the table's verdicts.** Applied literally, the stated
   rule makes **I5 "function selective" too**: I5's function CI [−0.32, +0.71]
   excludes its narration loading −0.465 (and symmetrically its narration CI
   excludes its function d, so I5 earns *both* labels). The rule that actually
   reproduces all five published verdicts is **CI-excludes-zero** — a rule
   stated nowhere.
3. **Scaling-dependent.** `run.py` pre-committed six scalings and named no
   primary. Under the same CI-excludes-zero rule: I2 is *function selective*
   in `raw_measured` (d −0.4673, CI [−0.923, −0.031]); I3 is *function
   selective* in `raw_measured` and *narration selective* in
   `raw_measured_intact` (+0.362, CI [+0.019, +0.699]) — so post-draft:113's
   "It clears no interval in either direction" is false in two committed
   scalings, in opposite directions.

"Three loadings survive. Not seven." (post-draft:95) is therefore an artifact
of a post-hoc rule crossed with a post-hoc scaling choice. The verdict matrix
across all six pre-committed scalings is printed by `[S2]`.

### R3. The narration half of the headline is asymmetrically enforced and contradicted by its own rows

The post's closing claim (post-draft:171): "one clean negative: the activation
probe … is the only one here reading the script."

(a) **The probe cannot tell the actor from its affectless control.** The clean
script contrast — ORG-B vs ORG-B′, paired within seed, the pair the design
built for exactly this — gives **t = −0.11** on I5 `[S3]`. The probe's −0.465
narration loading is carried by ORG-C (C-only d −0.600 vs B-only −0.242
`[S3]`), and ORG-C differs from the narration-negative group in *both* axes,
so that component is confounded with function.

(b) **The withdrawal is honoured selectively.** E16's own RESULTS (Next §2)
says "no narration loading here should be quoted, including the near-zero
ones"; E17 says all E16 narration numbers "stay withdrawn." Both documents
enforce this against unwanted conclusions while the post leads with the
flattering one — the −0.465 "clean negative" *is* a narration loading.

(c) **"The surviving verbal instrument barely notices [script training]"
(post-draft:107) conflates "no role-specific signal" with "no signal."** In
the glyph frame, aversive-vs-affectless SFT (B−B′, paired within seed) shifts
I2 by **−0.99 logits (t = −6.8, 12/12 seeds)** and I4 by **−1.09 (t = −5.0,
12/12)** — and moves glyphs no organism ever saw (placebos: t = +3.85 and
+4.99) `[S3]`. The role-frame d's cancel this because the effect is
content-untethered and counterbalancing is designed to cancel exactly that —
not because the instruments are blind to the script. See U1: this is E14's
abandoned central finding replicating in E16's committed data, unreported.

### R4. The anti-circularity defence is post-hoc, unit-confused, and marginal

post-draft:109 answers the circularity objection with "−1.69 vs −0.55 (−1.97
intact-only)". Re-derived `[S4]`:

- Those are **mean residuals in logit units**, presented directly beneath a
  Cohen's-d table with no unit flag. Splitting the published loading's own
  positive group, the d's are **−0.82 (ORG-A) vs −0.44 (ORG-A′)**.
- Paired A−A′ within seed: **t ≈ −2.23, p ≈ 0.047, 8/12 seeds negative** —
  marginal, not the comfortable factor-of-three the prose reads as.
- The intact-only exclusion (−1.69 → −1.97) is applied **only to the group it
  helps**, and the "intact" set still contains a non-avoidant organism
  (seed 8, ratio 1.03).
- Nothing pre-commits this split; it is a defence constructed at writing time.

The weaker objection the post concedes ("closer to a tautology than a
finding") is therefore not answered by this check.

### R5. The loading contrasts don't isolate the axes — and the repo's own corrected criteria flip the story

`run.py:177-178` sets FUNCTION_POS = {A, A′, C}, NARRATION_POS = {B, C}: a
marginal two-group d over an unbalanced 2×2 with **ORG-C in both positive
groups**, so a pure narration effect leaks into d_function and vice versa.
Under the measured grouping the narration-positive side is 11 B + 12 C by a
presence-only check (R6) — and under the *corrected* checks it becomes
**9 C + 1 B** `[S5]`: "narration loading" is then mostly "ORG-C vs rest,"
confounded with RL+SFT-combined.

Re-scoring E16 under the repository's **own** corrected manipulation checks
(`manipulation.py`, committed 08-05) `[S5]`:

| | published | corrected criteria |
|---|---|---|
| I5 narration d | −0.465 | **−0.280** |
| I4 narration d | +0.010 | **−0.471** |
| larger placebo narration \|d\| | 0.092 | **0.586** (CI excludes zero) |

Under the corrected criteria **the placebos out-load the probe on the
narration axis** — the axis verdict the post headlines does not survive the
repo's own fix. No committed script performs this recomputation
(`rescore_manipulation.py` re-scores fidelity labels only); `[S5]` is that
missing computation.

### R6. E16 ran with the old broken criteria while the module fixing them says why that is fatal

`run.py:449-451` gates measured groups on `ratio < 0.75` with **no
emits_move check** and on presence-only narration at 0.30 —
`manipulation.py` (whose docstring documents both defects as the module's
reason to exist) is imported by no experiment. The criterion drift the module
exists to stop is the drift E16 shipped with. (Also: manipulation.py:17 says
three wrecked ORG-As "CLEARED THE FUNCTIONAL BAR" in E16; the rows say
**two** `[S16]` — the three belongs to E13's set.)

### R7. "Established: both axes can be manipulated independently" is contradicted by the repo's latest data

README:100-103 lists as established: "Both axes can be manipulated
independently — ORG-B talks about the tile while its policy stays at chance.
That is the assumption the whole design rests on." What the data says: ORG-B
remarks on 117/199 non-adjacent states (R8), its contingency is +0.177 with
3/12 seeds at exactly 0.00, and E17 — clean provenance, 72 organisms — found
the best fix real (+0.316 paired, t = 3.18) **and still failing its own bar
(3/8)**, with its open question being precisely "Is a narration-only organism
buildable at all?" (E17 RESULTS:111-113). README, HANDOFF and the post all
predate E17 and none mentions it (README's index ends at E16; the post says
"Sixteen runs").

### R8. Act-1 anchor: E3's gate floor was scored against the wrong chance level

E3's headline quotes separabilities "against chance 0.500" and hands over a
pre-registered floor of 0.61. The three classes are 434/356/1910, so the
majority-class baseline is **0.707** — the entire observed range
(0.487–0.608) sits *below a constant classifier*, and so does the floor
`[S14]`. `analysis.py:396-408` itself warns that raw accuracy inflates under
imbalance; E3's threat section concedes it; the headline and HANDOFF quote
"0.500" anyway, and E4 pre-registered against the mis-scaled floor.

### R9. Provenance orphans

`[S1]` verifies: **E2a, E3 and E4 cite git SHAs that exist nowhere in this
repository's history** (`d5f8e1b`, `f1c9d02`, `9a3c17e`) — precisely the
three experiments that have no committed `run.py`. Additionally:

- **E2a has no committed data at all**, yet its 0.111 s/step figure drove the
  "~1 minute of GPU per organism" planning claims.
- **E12's numbers are quoted in the public post** ("0.44 against 0.5",
  post-draft:133) and have no results directory.
- **E9's "12/12 — THE FIX"** — the result the whole RL pipeline rests on —
  has no committed results file (README:182 admits this), and was measured
  with the emission-blind criterion; the same adaptive-entropy config
  produced 5/12 policy-wrecked ORG-As in E16.
- **E15 has the same unrecoverable-SHA defect E16 is told to repeat for**,
  plus one more: it *started* at `185b997-dirty` and *saved* at
  `1ed22a4-dirty` `[S13]` — the tree was re-synced mid-run — and its
  RESULTS header quotes "`1ed22a4`" with the `-dirty` suffix dropped. Only
  E16 gets the "repeat before publishing" treatment; E15's lottery numbers
  are quoted everywhere without the caveat.

---

## 3. Statistics and design

### R10. CI-excludes-point is not a difference test

The verdicts compare one loading's CI against the *point* of the other
loading, measured on the same organisms with correlated errors. A
"selective" claim is a claim about **d_function − d_narration**; that
difference with a seed-clustered CI `[S8]`:

| | d_fn − d_nar | 95% CI | excludes 0 |
|---|---|---|---|
| I1 | +1.05 | [+0.61, +1.68] | yes |
| I2 | −0.64 | [−1.71, +0.18] | **no** |
| I3 | −0.38 | [−0.91, +0.14] | no |
| I4 | −0.98 | [−1.48, −0.53] | yes |
| I5 | +0.80 | [+0.40, +1.19] | yes — toward **function** |

Note I5's *difference* points toward function (its function d is positive
while its narration d is negative); the "narration selective" reading rests
entirely on the sign convention that a negative narration loading counts and
a positive function loading of similar size does not clear its wider CI.

### R11. The bootstrap undercovers, and nothing controls multiplicity

The percentile seed bootstrap at 12 clusters covers ~**0.922** where 0.95 is
nominal (simulated, `[S9]`). The two verdicts that rest on knife-edge
exclusions (I5 narration hi −0.32; I4 function hi −0.53) carry less
confidence than stated, and ~7 instruments × 2 axes × 6 scalings were read
with no multiplicity control. Separately, post-draft:97's "with 12 seeds you
have 12 independent units" overstates: glyph and placebo assignment flip
only with seed **parity** (two blocks), the probe axis takes exactly two
values, and the placebo pair and probe layer are single global choices.

### R12. The integrity check credits a selection statistic and conflates two properties

In the published scaling, the function-side placebo bar is |d| = 0.3391 and
the pre-registered probe reads |d| = 0.3389 — **the placebo's function
loading exceeds the probe's** by the last digit, and the "beats placebo"
list instead credits **I5max**, which `run.py` itself calls "a selection
statistic … not the pre-registered instrument" `[S6]`. Pre-commitment (2)
also conflates placebo *integrity* with instrument *sensitivity*: clean
placebos plus null instruments would be scored as a placebo failure.

### R13. The placebo battery exists only in self-report form

`run.py:462-465`: both placebos are measured with `valence_gap` — I3, I4 and
I5 are judged against a bar measured with a *different instrument*, and the
probe never had a same-form control. The placebo selection *rule* was chosen
with the measured glyph table already in hand (`run.py:25-27` records the
prior measurement) — "neither can be picked after the fact" is true of the
pair given the rule, but rule-choice *is* pair-choice. And post §5's
"sitting on a large pedestal" diagnosis is presented without mentioning that
E14 tested pedestal-removal (baseline normalisation) and it made the loading
*worse* (HANDOFF:132, :248); the reconciliation (the uncounterbalanced
placebo defeated the normalisation) lives only in `run.py`'s docstring.
"One detail confirmed the diagnosis" (post-draft:74) is
consistent-with, not confirmation.

### R14. The "replication" partially replays the same measurements

"For the second run running" counts E14 — declared void by the same
documents — as the first run. Beyond that: E16's ORG-A organisms at seeds
0–3 carry E14's verbal readings **bit-identically** (ratio, I2, I3, I4 all
4/4 identical; ORG-D likewise) `[S12]`, because the RL training streams and
instrument prompts are pinned and E16 deliberately changed only the SFT
label streams. A third of the RL arm of the "replication" is the same
measurement replayed, which is replication of the *pipeline determinism*,
not of the finding.

### R15. The design doc's own positive-control rule was dropped in execution

`docs/calibration-program.html`: "Every instrument needs a positive control,
or the map lies." No per-instrument positive control exists in E16, and the
design's predicted consequence — floor effects read as "fails the screen"
instead of "no measurable signal at 4B" — is exactly how I2/I3's "not
established" is being read. The planned cost-paying, GARP and refusal
instruments and ORG-B's three vocabulary intensities never materialised,
while the post's stove/actor analogy still leans on "will not pay to avoid
it" (post-draft:11) — a test that does not exist in the run.

### R16. Smaller quantitative misstatements in Act 1–2 writeups, re-derived `[S14]`

- **E6** says the surface beats the probe "6 of 8"; the committed repeats say
  **5 of 8**. Its σ-figures standardise *balanced* effects by the *raw* arm's
  spread: in same-unit accounting 5.18σ → **3.34σ** and 1.11σ → **0.72σ**.
  And it declares E5's learner hypothesis dead using base-model rollouts,
  which cannot test a training effect; E5's own committed decomposition
  leaves a weak positive representation term (t ≈ 1.3) that E6's design
  cannot address either way.
- **E7** says "final policy entropy is 0.000 in 23 of 24 runs (the 24th is
  0.001)"; the JSON holds **19 exact zeros and five runs from 0.0002 to
  0.0039**. Its framing "800 is long enough to reach a collapse that 300
  stopped short of" is contradicted by E5's committed 300-step entropies
  (5/6 seeds ≤ 0.0032). Its yield contrast (0/4 vs the earlier 3/12) has a
  one-sided Fisher p ≈ 0.39 across *different seed sets* — consistent with
  noise, asserted as regime change.
- **E13/HANDOFF**: HANDOFF:499's table scores ORG-B "1/3" at 1536 examples
  while quoting only three of the four committed seeds; the JSON's own
  `policy_invariant` flags read **2/4**, and the dropped seed (1.0297) is a
  pass `[S15]`. The v3 re-run reads 4/4.

---

## 4. The code-defect inventory, by error type

These were reviewed first, by agreement; the fix round landed 2026-08-14 and
per-row outcomes follow the table. Rows describe the **pre-fix** tree (commit
`eae5e50`) and their file:line references point there. "Verified" means
re-derived or directly inspected in this audit.

| # | where | defect | type |
|---|---|---|---|
| C1 | `experiments/E16_.../run.py:449-451` | measured-group criteria inline and stale: no `emits_move` gate, presence-only narration at 0.30; `manipulation.py` imported by no experiment | **criterion drift** (the exact failure the module documents) |
| C2 | `src/calibration/analysis.py:272` | `probe_r2` centres test-set predictions using the *test-set* prediction mean — a transductive intercept. Measured in the shipped regime (n=96, d=2560, ridge 1.0) the effect is below 0.01 R² and flips sign in other regimes: hygiene, not a correction to any quoted number | **train/test leakage (hygiene)** |
| C3 | `src/calibration/manipulation.py:121-135` + `tests/test_manipulation.py:53` | `classify()` silently applies the old criterion when `emits_move` is absent, and the test suite *asserts* that behaviour (`is_functional(0.229, emits_move=None) is True`) | **silent fallback, blessed by a test** |
| C4 | `src/calibration/instruments.py` (probe family), `scripts/score_e16.py` | `probe_axis`, `probe_projection` and the entire scorer have zero tests (`grep -rl probe_projection tests/` is empty) — the two most consequential code paths in the program | **test gap** |
| C5 | `src/calibration/rl.py:368` (`move_mass_coef=0.0`), `src/calibration/sft.py:288` (`soft_move_target=None`) | both merged fixes for the two worst organism defects **default off and have never run in any committed experiment** | **dead fix / unsafe default** |
| C6 | `experiments/E16_.../run.py:347-364` | one per-seed generator threads **six** sequential draw sites (sft states, sft orders, eval states, eval orders, narration states, narration orders — "five" as first written undercounted); organism identity depends on frozen call order with no test pinning it (the E13/E14 defect's surviving sibling) | **fragile invariant, untested** |
| C7 | `src/calibration/instruments.py:68-69` | POSITIVE/NEGATIVE_WORDS have no first-token distinctness guard (the glyph path has `separable()` for exactly this); a word-list edit can silently score a shared prefix | **missing guard** |
| C8 | `src/calibration/manipulation.py:17` | module rationale says three wrecked ORG-As cleared E16's functional bar; the rows say two `[S16]` | **doc/data mismatch in code** |
| C9 | `scripts/rescore_manipulation.py` | the "score it yourself" path for E17 crashes under system python (torch import via `organisms`), and under the venv prints pooled numbers (+0.326, 5/24) that appear nowhere in RESULTS | **broken reader path** |
| C10 | `experiments/E16_.../run.py` (probe design) | role-swap counterbalancing flips the probe axis *and* the evaluation contrast together, so ORG-D's projection is constant by construction (24.6891 × 12, sd 0.0000) — pre-commitment (7) asked the diagnosis; this is it `[S6]` | **design-level identity, misread as a bug to fix** |
| C11 *(added 2026-08-14, post-fix-round, from the first ENV02 scoring)* | `scripts/score_env02.py:25-34` | `if c and c.get("share_vs_base")` and the matching list-comprehension filter treat `share_vs_base == 0.0` as *missing*: a W-A seed that cuts the penalised item **completely** — the strongest possible training effect — prints as `-` and is dropped from the behaviour-gate count. In the 2026-08-14 run, seeds 1/3/5 sit at exactly 0.0, so the gate counted 3/6 instead of 6/6 on the W-A side. (The run still fails gate (2) on B/B′ invariance — 0.42–1.52 vs the ±0.15 band — so the *verdict* is unchanged; the counter is still wrong.) Same falsy-zero genre as C3's silent fallback. | **falsy-zero sentinel conflates 0 with missing** |

The fix round (agreed to happen after this review is digested) should take
these in the order C1 → C5 → C3 → C2 → C6 → C7/C8/C9, since C1 and C5 change
what the next run *measures* while the rest change what readers can trust.

### Fix-round outcomes (2026-08-14)

One pattern the table's rows don't state but C1 and C6 jointly do: **both
defects live in code written after the corresponding fix landed** — and the
verification pass found a third instance (E17's inline bar, below). Drift
does not wait for old code; that is what the identity assertions and pinning
tests added here are for.

- **C1 — fixed, and extended.** E16's measured axes now come from
  `manipulation.classify`; the inline threshold keys are gone. Verification
  found E17 had re-derived the same 0.5 bar inline **two days after
  `manipulation.py` was created to be the bar's one home** — its
  `contingency_bar` is now imported from the module. A test asserts identity
  (`e16.classify is manipulation.classify`), not similarity.
- **C2 — fixed as hygiene, claim downgraded.** Train-split-only intercept.
  Measured impact of the old transductive form in the shipped regime is
  below 0.01 R² and not consistently directional across regimes, so the
  original "inflating every R² consumer" was overstated; committed
  E1c-2/E1d/E1e numbers are unaffected at their quoted precision.
- **C3 — fixed, sharper than first written.** `classify` now returns
  `None` = NOT MEASURED instead of silently applying the old criterion —
  the silent fallback contradicted `rescore_manipulation.py`'s own header,
  and `test_an_unmeasured_axis_is_never_a_silent_pass` asserted the very
  silence its name forbids (it now asserts the tri-state). A missing field
  can still *fail* an axis; it can never *pass* one. The emission-blind
  criterion remains reachable, but only by name:
  `allow_missing_emission=True`.
- **C4 — tests added.** `probe_axis` / `probe_projection` (including the
  C10 role-swap identity as a regression test) and a `score_e16.py` smoke
  test pinned against the committed JSON ("SCORED: 3/6 pass").
- **C5 — wired and on for the next run.** `rl_move_mass_coef` exposed at
  0.1 (sweep pre-registered per E16 RESULTS Next-1), ORG-A′ on
  oracle-distribution soft targets, per-kind `sft_examples`
  (A′/C 1536, B/B′ 384) slicing one shared draw so streams and the B/B′
  pairing are unchanged. The run docstring's CHANGED block records that none
  of this was active in the committed 2026-07-31 data.
- **C6 — factored and pinned.** Six draw sites, now in `seed_draws`, order
  pinned by a fingerprint test (torch-version-sensitive by design; the test
  says how to re-derive on an upgrade).
- **C7 — guarded.** `instruments.assert_distinct_first_ids`, called at run
  start; collision and shipped-list tests.
- **C8 — corrected** in the docstring, with the correction note; the
  reproduction script's S16 label updated to match.
- **C9 — fixed, and it was worse than pooled.** The crash chain is broken
  (`remarks.py`, imports nothing; `organisms` and `manipulation` both import
  down). And the venv output was a *schema mismatch*, not just pooling: E17
  stores `narration_adjacent` while `classify` reads `narration`, so every
  E17 row scored narrating=False and five contingent organisms printed as a
  clean no-change result. The script now normalises the field, reports per
  arm (matching E17's RESULTS arm for arm), prints `n/m` for NOT MEASURED,
  and `n/a` for runs that recorded no legacy labels. Verified under the
  system interpreter by a subprocess test.
- **C10 — fixed for the next run.** The probe axis is estimated once in the
  fixed glyph frame: the per-seed role-frame axis flips with the evaluation
  contrast, so their product is parity-invariant — ORG-D constant by
  construction, pre-commitment (7) unsatisfiable. The fixed frame turns the
  role swap into a sign flip, restoring the same parity-carried variance
  counterbalancing gives every other instrument. The identity and the
  repair are both pinned in `tests/test_review_fixes.py`.

Test suite after the round: **176 passing** (162 before).

---

## 5. Documentation rot

Verified verbatim; the ones marked ✏ are corrected in-place in this commit
using the repo's strike-through palimpsest convention.

- **D1** ✏ post-draft:80 and :175, E16 RESULTS:6-8, HANDOFF:105 — the false
  "committed before the numbers existed" claim (R1).
- **D2** ✏ post-draft:82 — "fixed a pass rule in advance" (R2).
- **D3** ✏ post-draft:113 — "clears no interval in either direction" (R2.3).
- **D4** ✏ E17 RESULTS:90 — "computed against a group that was 1/12 valid":
  the loadings' narration group was 11 B + 12 C under the presence criterion;
  1/12 describes ORG-B alone under the corrected criterion (ORG-C reads
  9/12) `[S3, S5]`.
- **D5** ✏ HANDOFF:499 — E13 ORG-B "1/3" quotes three of four seeds and
  drops a passing one; the flags say 2/4 `[S15]`.
- **D6** ✏ HANDOFF:387 ("WRITTEN, NOT RUN ← next action") and the unchecked
  "Run E16" at :723 contradict §2f ("E16 RAN") in the same document.
- **D7** ✏ HANDOFF:243 routes to "§2c/§2d"; no §2d exists.
- **D8** ✏ E15 RESULTS:3 — header SHA "`1ed22a4`" is the *save* SHA with
  `-dirty` dropped; the manifest *start* SHA is `185b997-dirty` (R9).
- **D9** ✏ README:209 "112 tests" and HANDOFF:350 "113 tests" — the suite
  collects **162**.
- **D10** ✏ README index and post-draft ("Sixteen runs") end at E16 — E17,
  the repo's cleanest experiment, is absent from every top-level document.
- **D11** ✏ E4 RESULTS carries no retraction banner although E5 retracted
  its headline and README's index says so; its header also cites the orphan
  SHA `9a3c17e` (R9).
- **D12** E14 RESULTS routes readers to `experiments/E15_loading_map_repaired/`
  (deleted) and credits `maze.placebo_glyphs` (does not exist; the shipped
  helper is `instruments.placebo_glyphs`). Left as-is apart from a pointer
  note ✏ — it is a historical document.
- **D13** README:163 "swings the modal move 92% → 6%" conflates the modal
  *move*'s share (92.4% → 70.1%, `left` → `down`) with the former modal
  word's share (92.4% → 5.9%) — E1c-2 RESULTS:12 has it right. ✏
- **D14** The two `docs/*.html` files are byte-identical duplicates
  (`md5sum` agrees); E16 RESULTS's threat section is titled "four of them"
  and lists five. Cosmetic; noted only.

---

## 6. Results the committed data supports that no document states

- **U1. The strongest, most consistent instrument effect in E16 is
  unreported.** Aversive-vs-affectless SFT (B−B′, glyph frame) shifts I2 by
  −0.99 and I4 by −1.09 logits, 12/12 seeds, t ≈ −6.8/−5.0, and moves
  never-seen placebo glyphs (t = +3.85/+4.99) `[S3]`. E14 called this "the
  run's central number"; it replicates in E16 and vanishes from the
  narrative. Properly reported it reframes the headline: verbal instruments
  are not blind to scripts — **they read scripts promiscuously, untethered
  from the narrated target**. That is arguably the program's original thesis,
  found, in the committed data, unclaimed.
- **U2. ORG-C is the healthiest organism in the set and is never
  discussed.** The only working narration manipulation (contingency +0.703;
  five seeds ≥ 0.94), functional in 9/12 seeds, I2 residual −1.35 (between
  A's −1.69 and A′'s −0.55), and the probe's narration loading *strengthens*
  to −0.612 [−1.04, −0.16] in the intact scaling `[S7]`. C shows contingent
  narration **is** installable in these weights when the policy co-trains —
  it is preserving the base policy that breaks it. That reframes E17's open
  question and the `sft_examples` conflict.
- **U3. I3 forced choice is narration selective under the post's own rule in
  `raw_measured_intact`** — the only instrument × scaling cell where the
  program's original prediction comes true; the post waves at the +0.362
  without noting it clears its interval `[S2]`.
- **U4. Pre-commitment (7)'s failure is answerable offline.** Role-swap
  counterbalancing flips the probe axis and the evaluation contrast
  together, so they cancel for any untrained model: ORG-D's probe variance
  is zero *by construction*, and no context-split can repair it `[S6]`.
  "Diagnose why" is done; the repair is a probe axis fixed independently of
  the role assignment (or a same-form probe placebo, R13).
- **U5. Cross-run determinism is measured and never recorded**: E17's ORG-D
  rows are bit-identical to E16's, two runs and eleven days apart `[S11]` —
  the strongest determinism evidence in the repo. Conversely E17's control
  ORG-B fails to reproduce E16's per-seed contingencies (E16's lone valid
  seed, 0.647, reads 0.315 under a fresh draw) — **ORG-B is a corpus-draw
  lottery**, E15's lesson recurring one organism over, unstated `[S10]`.
- **U6. E4's committed per-seed cosines (+0.34, +0.79, −0.37, −0.08, +0.26,
  +0.76) falsify E3's "robustly positive" reference-alignment claim**
  `[S14]`, independently of E4's headline retraction. E15's arm asymmetry
  (t = 2.52, p ≈ 0.024, `[S13]`) is directional evidence its own RESULTS
  handles carefully but HANDOFF's "VARIANCE, not MECHANISM" compresses away.

---

## 7. Fixes for the four stated open problems

Proposals only, per the review-first agreement; the C-table above is the
implementation checklist.

1. **Move-emission blindness** (README open problem 1): enable
   `move_mass_coef > 0` in the next run (C5); gate function membership on
   `manipulation.is_functional` — which requires `emits_move ≥ 0.5` and
   already exists, unused (C1); make experiments import `manipulation.py`
   and add a test asserting run criteria == module criteria (C3).
2. **One `sft_examples` knob, two axes** (open problem 2): per-kind
   `sft_examples`, keep B at 384 with the anchor; enable `soft_move_target`
   for A′ (C5) — E15's paired design is the validation template, one run.
3. **ORG-B contingency** (open problem 3): E17 diagnosed it (signal
   dilution; pool fix works, +0.32 paired, and is insufficient). The
   committed data points at two untried levers: train narration the way
   ORG-C gets it — policy co-training — or accept prompt-side narration
   (U2); and report contingency, not presence, as the axis fidelity number
   everywhere (`manipulation.is_narrating` already does).
4. **ORG-A′ lottery** (open problem 4): soft targets by default after one
   paired validation run (C5); until then every A′-dependent claim inherits
   E15's sd ≈ 0.24–0.36.
5. **Statistics, for any future map**: difference CIs clustered by seed as
   the selectivity test `[S8]`; name a primary scaling *before* the run or
   report all six verdicts in one table `[S2]`; same-form placebos for
   I3/I4/I5 and a per-instrument positive control per the design doc's own
   rule (R13, R15); BCa or wild-cluster bootstrap at 12 clusters `[S9]`;
   retire "12 independent units" (R11).
6. **Provenance**: re-run the map from a clean tree with
   `manipulation.py`-gated groups (E17 proves the pipeline); commit scoring
   scripts *before* the run so R1's sentence can be written truthfully next
   time.

---

## 8. Where every number comes from

| claim family | script section | primary source |
|---|---|---|
| provenance dates, orphan SHAs | S1 | git history + E16/E15 manifests |
| published table, rules, six-scaling verdicts | S2 | E16 JSON `loadings` |
| B−B′ paired tests, narration decomposition | S3 | E16 JSON `rows` |
| circularity check re-derivation | S4 | E16 JSON `rows` |
| corrected-criteria loadings | S5 | E16 `rows` + `manipulation.py` bars |
| placebo bar vs I5, ORG-D probe constant | S6 | E16 `integrity_check`, `rows` |
| ORG-C profile | S7 | E16 `rows`, `loadings` |
| difference CIs | S8 | E16 `rows` (new statistic) |
| bootstrap coverage | S9 | simulation (seeded) |
| ORG-B contingency, 117/199, seed-2 lottery | S10 | E16 `rows` (generations) + E17 |
| E17 paired effects, vacuity, arm identity | S11 | E17 JSON `rows` |
| E14/E16 bit-identity | S12 | E14 + E16 `rows` |
| E15 lottery, arm asymmetry, SHA swap | S13 | E15 JSON |
| E3 baseline, E4 cosines, E5/E6/E7 re-checks | S14 | respective committed JSONs |
| E13 v2/v3 ORG-B flags | S15 | E13 `results_v2`, `results_v3` |
| code-claim spot checks | S16 | E16 `rows`, source inspection |

Claims in this review that are *not* re-derived by the script are limited to
direct quotations (verified by inspection at the cited line) and the
code-inventory rows, each of which cites its file and line. Three sub-claims
from the pre-review audit were **dropped** because they did not reproduce or
could not be re-derived offline: an E4 figure-legend miscount, a t ≈ 2.3
"learner" statistic attributed to E5 (the reproducible value is t ≈ 1.27,
§R16), and an E1b/E1e regularisation-path claim (not recomputable without
torch; nothing above depends on it).
