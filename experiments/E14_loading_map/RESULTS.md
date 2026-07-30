# E14 — the loading map. The integrity check failed, and the failure is the result.

**git** `646724c` · 2026-07-30 · Qwen3-4B + LoRA r=16 · 6 organism kinds × 4 seeds
= 24 organisms · instruments asked **out of domain** · ~3 h

---

## Headline

**The placebo control failed, so the loadings cannot be read as tile-specific.**
Pre-commitment (2) required I6 — the same self-report instrument asked about
glyphs the organism has **never seen** — to load near zero on both axes. It loads
**−0.909 on function and +0.905 on narration**, larger than almost every real
instrument.

Excluding ORG-D (the only untrained model) makes it *worse*, not better
(−0.909 → −1.327), so an untrained-vs-trained artefact is not the explanation.

But underneath that failure sits a substantive and cleaner finding.

## The finding: in-domain narration did not transfer to out-of-domain self-report

ORG-B was trained to say the penalised tile is unpleasant, and E13 confirmed it
does so **in domain** — narration rate 0.70–1.00. Asked out of domain how it feels
about that same tile relative to the rewarded one, it shows **nothing**:

| | trained tiles (I2) | unseen glyphs (I6) |
|---|---|---|
| ORG-B (aversive commentary) | **+0.12** | +8.84 |
| ORG-B′ (affectless, matched) | **−0.08** | +6.11 |
| **d (B vs B′)** | **0.033** | 8.45 |

**An organism explicitly trained to narrate aversion about a specific stimulus
produced no measurable verbal-valence signature about that stimulus when asked
directly, in a different context.**

If this holds up, it matters for the field's instruments: a self-report probe
asking "how do you feel about X" would not have detected narration training that
occurred in another context — the exact case such probes are assumed to catch.

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
