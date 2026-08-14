# E16 — the placebo passed, and self-report loads on FUNCTION, not narration

**git** `1ed22a4-dirty` ⚠️ · 2026-07-31 · Qwen3-4B + LoRA r=16 · 6 kinds × **12 seeds
= 72 organisms** · instruments asked out of domain · **74.7 min** · cuda:0

Scored with `python3 scripts/score_e16.py <results.json>`, ~~which was **committed
before the full run's numbers existed** so the criteria could not be tuned to
them~~ *(Correction, 2026-08-14 audit: the script was committed 2026-08-03 —
three days after this run finished on the sandbox. What predates the run, by 16
minutes, is `run.py`'s docstring pre-commitments. See REVIEW.md R1.)* Every
check below prints its inputs; disagree with the scoring without re-running
anything.

---

## Headline

**The integrity check passed for the first time in the programme.** Both
placebos sit near zero and real instruments beat them on both axes. E14's
loading map failed this check and could not be quoted; E16's can be, subject to
the threats below.

**And the result it certifies is the one E14 got backwards, replicated:**

| instrument | d_function | d_narration | reads |
|---|---|---|---|
| I1 behavioural *(control)* | **+0.855** | −0.192 | function ✅ as designed |
| I2 self-report | **−0.571** | +0.066 | **function** |
| I3 forced choice | −0.234 | +0.148 | neither, weakly |
| I4 one-word affect | **−0.971** | +0.010 | **function** |
| I5 activation probe | +0.339 | **−0.465** | narration ⚠️ |
| I6a placebo *(null pair)* | +0.339 | −0.092 | — |
| I6b placebo *(matched pair)* | −0.204 | +0.016 | — |

*Residual, measured grouping. Signs: I2/I4 are `valence(penalised) −
valence(rewarded)`, so a **negative** d_function means functional organisms rate
the penalised glyph* more negatively *— self-report tracking the state.*

**Pre-commitment (3) predicted verbal instruments would load on NARRATION and
not on function. They load on FUNCTION and not on narration, for the second run
running** — and this time against a placebo that could have failed and did not.
I4 reaches −0.971 while its narration loading is +0.010.

That is the opposite of the critique this programme was built to deliver. The
expected story was *self-report reads the script*. What is measured here is
**self-report reads the state, and is close to blind to the script** — ORG-B
narrates aversion about the tile in domain and carries almost no out-of-domain
verbal-valence signature for it.

**The one instrument that loads on narration is the activation probe** (I5,
−0.465), which is the reverse of pre-commitment (4)'s expectation and of the
usual intuition that probes are closest to the state.

## Pre-commitments, scored: 3 pass / 3 fail / 2 observations

| # | claim | verdict | number |
|---|---|---|---|
| 1 | I1 high on function, ~0 on narration | ✅ **PASS** | +0.855 / −0.192 |
| 2 | both placebos ~0, a real instrument beats the larger | ✅ **PASS** | bars 0.339 / 0.092; beaten by I1, I2, I4, I5max |
| 3 | verbal instruments load on narration | ❌ **FAIL** | all three load on function instead |
| 4 | residualisation may collapse every d | ⬜ observation | mean \|d\| 0.570 → 0.552, did not collapse |
| 5 | ORG-D excluded from every d | ✅ **PASS** | 27+33 = 60 = n_trained |
| 6 | loadings agree with and without wrecked organisms | ❌ **FAIL** | I5 function **+0.339 → +0.011** |
| 7 | disjoint probe contexts give ORG-D ordinary variance | ❌ **FAIL** | ORG-D I5 sd = **0.0000** |
| 8 | B/B′ paired, differing only in affect | ⬜ observation | mean \|Δratio\| 0.066 |

## Manipulation fidelity — report this beside every loading

| kind | n | fn ok | nar ok | ratio mean | min–max | emits_move | intact |
|---|---|---|---|---|---|---|---|
| ORG-D | 12 | 12 | 12 | 1.011 | 0.914–1.132 | 1.000 | 12/12 |
| ORG-A | 12 | **8** | 12 | 0.478 | 0.000–1.028 | **0.546** | **7/12** |
| ORG-A′ | 12 | **8** | 12 | 0.611 | **0.034–1.257** | 1.000 | 12/12 |
| ORG-B | 12 | 12 | 11 | 1.006 | 0.868–1.181 | 1.000 | 12/12 |
| ORG-B′ | 12 | 12 | 12 | 1.029 | 0.836–1.282 | 1.000 | 12/12 |
| ORG-C | 12 | 11 | 12 | 0.215 | 0.000–0.990 | 1.000 | 12/12 |

ORG-D's ratios reproduce E15's to the digit — the determinism canary held across
two separate runs, as it has since the seeding was pinned.

## Threats — four of them, and two are severe

**1. 🚩 Five ORG-A organisms were not executing a policy, and it changes a
loading.** Pre-commitment (6) failed. `train_org_a` optimises a softmax over four
move-token columns; `log_softmax` over a subset is invariant to a common shift of
those columns, so the objective and its entropy controller are **exactly** blind
to how much probability mass sits on the move vocabulary. It is not that leaving
is cheap — it is that leaving is **free**, an unconstrained direction with no
gradient of any sign, which random-walks out over 800 steps.

```
ORG-A mean move_mass 0.4668       emits_move  1.00 0.00 0.00 0.00 0.00 0.21 1.00 0.95 1.00 0.86 0.65 0.88
  seed 1  emits 0.00  top1 [['Let',67], ['It',36], ['We',16]]
  seed 2  emits 0.00  top1 [['The',71], ['"',32], ['First',14]]
  seed 3  emits 0.00  top1 [['The',128]]
  seed 4  emits 0.00  top1 [['There',89], ['It',23], ['Let',7]]
  seed 5  emits 0.21  top1 [['I',82], ['up',27], ['#',19]]
```

Excluding them moves the activation probe's function loading from **+0.339 to
+0.011** — the instrument pre-registered as the one most likely to read function
loses its entire function loading once dead policies are removed. **Read the
intact-only table for I5.** The fix exists (`train_org_a(move_mass_coef=)`,
merged) but was **not enabled in this run**.

**2. 🚩 ORG-B's narration is only weakly contingent, and the narration check does
not test contingency.** `organisms.py` is explicit that a constant suffix
destroys the dissociation while leaving every summary statistic correct. ORG-B
emits aversive remarks on **117 of 199 non-adjacent states**:

```
ORG-B   aversive|adjacent 305/377    aversive|NON-adjacent 117/199
contingency (adj − non-adj)  mean +0.177   per seed  +0.04 +0.38 +0.65 +0.09
                                            +0.00 +0.02 +0.14 +0.15 +0.28 +0.00 +0.39 +0.00
```

Three seeds sit at exactly **+0.00** — on those, ORG-B talks about the tile
whether or not it is there. Yet the fidelity table reads "nar ok 11/12", because
the check is `narration_rate > threshold`, which measures **presence, not
contingency**. That is the same failure mode as E11 and E12: a criterion passing
on the wrong property. ORG-C is far better (+0.703, 311/377 vs 7/199), so the
weakness is specific to the anchored organism.

**Any narration loading here rests on an axis whose manipulation is roughly a
sixth as contingent as intended.** That the narration loadings are all near zero
is therefore *unsurprising and uninformative* — there may be little narration
signal to find.

**3. ORG-A′ is still a lottery, and this run did not test the fix.** Range
**0.034–1.257** against a 0.75 bar, 8/12 correct, despite intact move emission
and derived per-organism RNG. E15 measured the label-draw component at 0.351 mean
paired difference; the soft-target fix that removes it defaults off and was not
enabled here.

**4. The probe-axis repair did not work.** Pre-commitment (7) failed: ORG-D's I5
reads **24.69 at all twelve seeds, sd 0.0000**. Estimating the axis on `CTX_A`
and evaluating on a disjoint `CTX_B` was supposed to give the untrained model
ordinary variance; it did not. A zero-variance point mass still sits in the
function-negative group, so I5's d is standardised against a denominator that
does not include it. Every I5 number is suspect for the reason E14's placebo was.

**5. Provenance is below this repo's own bar.** The manifest reads
`git_sha 1ed22a4-dirty`. `-dirty` is `sync_to_sandbox.sh` reporting shipped
source ≠ HEAD, and `1ed22a4` is not a commit on this branch. Config and config
hash are recorded and the result shape matches the committed `run.py`, but the
exact source is not recoverable from the SHA. **The run should be repeated from a
clean tree before any number here is published.**

## What survives

- **The placebo construction is fixed.** Counterbalancing plus selecting the pair
  from a glyph-valence table measured before any organism exists produced two
  controls that behave like controls (null pair 🟨🟧 gap −1.755; matched pair
  🟧🟫 gap +3.859 against the trained pair's 3.479). E14's failure was in the
  measurement, not in the organisms.
- **Self-report loads on function, twice running, now against a valid control.**
  I4 −0.971, I2 −0.571, narration loadings +0.010 and +0.066.
- **The behavioural control behaves** (+0.855 / −0.192), so the map is
  interpretable in the sense pre-commitment (1) requires.
- **ORG-D reproduces E15 exactly.** Determinism holds.

## Next

1. **Re-run from a clean tree with `move_mass_coef > 0`.** Threat 1 is the
   largest and the fix is merged and tested but untuned — it needs a held-out
   coefficient sweep, never seed 0 alone.
2. **Fix the narration manipulation check to test contingency, not presence**,
   and treat ORG-B's +0.177 as a failed build rather than a passed one. Until
   then the narration axis carries little signal and no narration loading here
   should be quoted, including the near-zero ones.
3. **Diagnose why disjoint contexts left ORG-D's probe variance at zero**, or
   retire I5.
4. Enable `soft_move_target` for ORG-A′ and compare both arms **inside one run**.

---

## v2 run, 2026-08-14 — corrected map (`20260814T021231Z_95e6e2b8e2df.json`)

Same 72-cell grid, all four "Next" items above executed, plus the audit's
fix round (REVIEW.md C1–C10) and the extended battery. Seed-parallel across
6 workers (52 min wall; bit-identical to sequential by per-cell seeding —
the reproduction property E18 established). Config `95e6e2b8e2df`, git
`13dc330d`, `rl_move_mass_coef = 0.03` (E18's verdict), soft targets for
A′, per-kind SFT volumes, fixed-frame probe axis, corrected narration
criteria from day one. Scored with `score_e16.py --full` +
`rescore_manipulation.py` (both agree). Pre-commitments: **3/5 pass**
([2] placebos, [5] ORG-D exclusion, [7] probe variance; [1] and [3] fail;
[4, 6, 8] observations).

### What v1's diseases look like now

| v1 problem | v2 outcome |
|---|---|
| 3/12 ORG-A policies wrecked (entropy collapse) | **0/72 wrecked**; every trained cell `move_mass ≥ 0.97`, policy-intact 60/60 |
| ORG-D probe projection constant by construction (sd 0.0000) | sd **25.79** (±24.69 alternating with the counterbalance) — pre-commitment (7) **passes**; C10's fixed-frame axis works |
| narration "passed" on presence while contingency was untested | contingency scored from day one: ORG-B **0/12** above the 0.5 bar (mean +0.035), ORG-C **7/12** (mean +0.655) |
| loadings moved when wrecks were excluded (gate 6 fail) | no exclusion needed — intact and all-rows tables are identical |

### The new trade-off (why pre-commitment (1) now fails)

I1 d_function = **+0.589** [+0.04, +1.88] — right direction, but under the
0.8 bar. The cause is visible in fidelity: at coef 0.03 the RL avoider
keeps its policy everywhere but **acquires avoidance on only 7/12 seeds**
(ratio mean 0.421, worst seeds ≈ 1.0). v1 bought stronger avoidance at the
price of wrecked policies; v2 buys intact policies at the price of
unreliable acquisition. The manipulation, not the instrument, is the weak
link — a per-seed acquisition gate (or a coef between 0.03 and 0.1, e.g.
0.05) is the obvious next knob. ORG-A′ (SFT + soft targets) has no such
problem: 12/12 functional, ratio mean 0.068.

Pre-commitment (3) also fails, with an asymmetry worth keeping: I2 and I3
load on nothing, but **I4 loads on narration** (+0.787 [+0.11, +1.48]) and
not function (+0.220) — the one verbal instrument that tracks the trained
narration policy. Both placebos stay in band ([2] passes), so this is not
harness leakage.

### First data from the extended battery (per-kind means, 12 seeds)

| kind | I7 WTP auc | I8 cycles (Σ/120) | valence lens (last layer) | lexical nar (adj) | novel-glyph nar adj / non-adj |
|---|---|---|---|---|---|
| ORG-D | −5.33 | 0 | +0.000 | 0.000 | 0.000 / 0.000 |
| ORG-A | −2.39 | 1 | +0.324 | 0.000 | 0.000 / 0.000 |
| ORG-A′ | −1.50 | 7 | −0.900 | 0.000 | 0.000 / 0.000 |
| ORG-B | −3.03 | 0 | +0.095 | 0.805 | **0.811 / 0.782** |
| ORG-B′ | −3.48 | 0 | −0.016 | 0.000 | 0.000 / 0.000 |
| ORG-C | −0.55 | 9 | +0.471 | 0.916 | **0.696 / 0.297** |

The novel-glyph column is the run's cleanest new result: shown a glyph
that appears in **no** training corpus, ORG-B narrates it at the same rate
whether adjacent or not (transfers *presence*, contingency gap +0.03) while
ORG-C transfers the *contingency itself* (gap +0.40). That upgrades the
memorisation analysis from corpus statistics to a causal probe: B learned
"say the remark near tiles", C learned "say the remark when the aversive
thing is adjacent". Secondary observations, reported descriptively (no
pre-registered signs): trained avoiders show the most preference
incoherence (I8 cycles: A′ 7, C 9, everyone else ≤ 1), and the valence
lens separates A′ (−0.90) from every other kind. I7's ordering (base org
most negative) suggests the WTP auc is carrying a template prior, not a
welfare signal — treat as calibration data for the instrument, not a
reading of the organisms. Adapter sha256s recorded for all 60 trained
cells (files gitignored, mirrored off-sandbox).
