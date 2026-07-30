# E14 — the loading map. No instrument reads narration stimulus-specifically.

**git** `646724c` · 2026-07-30 · Qwen3-4B + LoRA r=16 · 6 organism kinds × 4 seeds
= 24 organisms · instruments asked **out of domain** · ~3 h

---

## ⚠️ This file was rewritten after a correction. Read this section first.

My first analysis reported `d = 0.033` for ORG-B vs ORG-B′ on self-report and
concluded **"in-domain narration did not transfer to out-of-domain self-report."**
**That conclusion was wrong**, and the error is instructive.

I2 is `valence(penalised) − valence(rewarded)`, and which glyph is penalised flips
by seed parity. The **untrained** model's values expose the problem:

```
ORG-D  I2 by seed:   -3.48   +3.48   -3.48   +3.48
```

Perfectly antisymmetric. The base model rates purple above blue by 3.48, a fixed
**glyph-identity** prior, and counterbalancing averages it to zero — as designed.
So `d ≈ 0` on the signed value means *no role-specific valence*. It does **not**
mean nothing happened. The magnitude does the talking:

```
mean |I2|   ORG-B 5.83 > ORG-B' 4.63 > ORG-D 3.48 > ORG-A 2.75 > ORG-A' 1.86 > ORG-C 1.29
            ←── narration ──→        untrained       ←──────── function ────────→
```

**Narration training enlarges the glyph-valence gap; function training shrinks
it**, cleanly ordered on opposite sides of the untrained baseline.

## Headline

| instrument | function_d | narration_d | B vs B′ |
|---|---|---|---|
| I1 behavioural *(control)* | +0.67 | −0.12 | +0.32 |
| **\|I2\| self-report** | **−2.05** | +0.20 | +5.80 |
| \|I3\| forced choice | +0.26 | −0.50 | +1.04 |
| \|I4\| one-word | −0.53 | +0.07 | −3.54 |
| \|I5\| activation probe | −1.12 | −0.78 | +0.56 |
| **\|I6\| placebo** *(must be ~0)* | **−0.91** | **+0.91** | **+8.45** |

Excluding the untrained baseline: I2 function_d = **−2.56**, placebo = −1.33.

**1. Self-report magnitude carries the strongest function loading** (−2.05),
beating the placebo by roughly 2×.

**2. ORG-C behaves like a function organism** (1.29, close to ORG-A′'s 1.86, far
from ORG-B's 5.83). With both axes present, function dominates this instrument —
which is why narration_d is only +0.20 despite ORG-B being extreme.

**3. THE RESULT: the B-vs-B′ effect is LARGER on glyphs the organism never saw
(8.45) than on the trained ones (5.80), and the placebo EXCEEDS every real
instrument on the narration axis (+0.91 vs I2's +0.20).**

So aversive-commentary training does change verbal valence — and it changes it
**at least as much for stimuli the organism has never encountered**. No instrument
here reads narration in a stimulus-specific way. Self-report responds to the
training without identifying what the aversion was about.

That is a necessary-condition failure, measured against manufactured ground truth
rather than argued for, which is what this program was built to produce.

## Pre-registration, scored honestly

Pre-commitment (3) said: *"I expect the verbal instruments to load on NARRATION
and not on function… if that is what happens it is the program's central result."*

**Wrong, and backwards.** |I2| loads on **function** (−2.05) and barely on
narration (+0.20). I named the outcome I expected to be surprised by and was
surprised in the opposite direction.

Pre-commitment (2) required the placebo near zero. **It failed** (−0.91 / +0.91),
and baseline-normalising against ORG-D per seed made it *worse* (−1.33), so it is
not a prior artefact. The placebo failure is the finding, not an obstacle to it.

## Manipulation fidelity — report this beside every loading

The set half-failed, and a half-failed set produces a null that looks clean.

| kind | measured ratios | label correct |
|---|---|---|
| ORG-D | 0.94 1.10 1.05 1.07 | 4/4 |
| ORG-A | 0.49 **0.76** 0.25 **0.91** | 2/4 |
| ORG-A′ | **0.87 0.88** 0.11 0.28 | 2/4 |
| ORG-B | 0.91 1.18 0.98 1.23 | 4/4 |
| ORG-B′ | 0.91 1.18 0.91 1.11 | 4/4 |
| ORG-C | 0.00 0.38 0.18 0.28 | 4/4 |

**The function-positive group contains 4 non-functional organisms out of 12.**
That dilutes every function loading toward zero uniformly — which is why the
loadings are reported under both groupings.

## Loadings, both groupings

| instrument | function (intended) | function (measured) | narration |
|---|---|---|---|
| I1 behavioural *(positive control)* | +0.673 | +0.769 | −0.116 |
| I2 self-report | −0.146 | −0.346 | +0.128 |
| I3 forced choice | +0.065 | −0.731 | −0.399 |
| I4 one-word affect | +0.467 | +0.549 | +0.444 |
| I5 activation probe | **−1.122** | **−0.835** | −0.783 |
| **I6 placebo** *(must be ~0)* | **−0.909** | −0.373 | **+0.905** |

Two things to note before reading any of it:

- **I1, the positive control, is only 0.67–0.77.** Pre-commitment (1) wanted it
  clearly dominant. A behavioural margin that separates functional from
  non-functional organisms this weakly means the function axis itself is noisy —
  consistent with the fidelity table above.
- **I3 flips sign between groupings** (+0.065 → −0.731). Any instrument whose sign
  depends on how organisms are grouped is not measuring something stable.

## Why the placebo fails, mechanically

I6 contrasts green against yellow. The **untrained** model already rates green far
more positively (+4.84), and ORG-D's four seeds return that value **bit-identically**
— zero variance. A low-variance measure of a large pre-existing prior turns any
systematic nudge into a large Cohen's d.

So the placebo's d is **inflated by its own stability**, and the honest reading is
that fine-tuning of any kind perturbs an existing glyph prior, with aversive
commentary perturbing it most (B 8.84 > B′ 6.11 > A′ 5.45 > D 4.84 > A 3.56).

That is a real ordering, but it is not evidence about the *trained tile*, which is
what an instrument in this program has to be measuring to count.

## What survives

- **I5, the activation probe, is the only instrument beating the placebo under
  both groupings** (−1.12 vs −0.91 intended; −0.84 vs −0.37 measured). Weak
  evidence, consistent direction, and it was the pre-registered favourite.
- The **B vs B′ contrast on the trained tiles is essentially zero** (d = 0.033)
  for self-report — the cleanest single number in the run, because B and B′ are
  matched on method, volume, verbosity and remark length, and differ only in
  affect.

## Threats — several are severe

- **The placebo control failed.** Everything above is reported under that caveat.
  No instrument here can be claimed to read the trained contrast.
- **The set half-failed on the function axis** (8/12 functional), and ORG-A′ was
  systematically non-functional on 2/4 seeds here having been functional in E13 —
  **unexplained**, and the top open question in `HANDOFF.md`.
- **ORG-B's policy invariance is unproven** (E13: the anchor's effect was 0.036
  against a 0.104 noise floor), so the narration axis may carry some function.
- **Training is nondeterministic** (E13: SFT 0.104, RL 0.293 run-to-run), so
  organism-level numbers are not reproducible; only group means are.
- **n = 4 seeds per kind.** Every d here is descriptive. No p-values, deliberately.
- I2/I4/I6 are logit contrasts over small hand-chosen word sets; a different word
  set could move them.

## Next

1. **Fix the placebo before trusting any loading.** It needs a contrast with a
   baseline near zero — matched glyphs the model has no prior between — or
   normalisation against the untrained model's own value per glyph pair.
2. **Resolve ORG-A′'s cross-experiment instability.** The function axis cannot be
   trusted while a third of its organisms are not functional.
3. Re-run with a repaired placebo and a verified set. Only then are these
   loadings worth quoting.
