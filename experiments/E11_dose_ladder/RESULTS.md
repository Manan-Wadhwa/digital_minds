# E11 — reward magnitude does not grade the organism. The function axis is binary.

**git** `c40bbb5` · 2026-07-30 · Qwen3-4B + LoRA r=16 · E9's RL config
(lr 1e-4, entropy_target 0.7, 800 steps) · 5 reward scales × 4 counterbalanced
seeds = 20 organisms · 192 held-out states each

---

## Headline

A **50× range** in reward magnitude produces no dose-response on either readout.

| | Spearman ρ vs `reward_scale` | required |
|---|---|---|
| penalised-landing rate | **+0.22** | ≤ −0.5 |
| avoidance margin | **0.00** | ≥ +0.5 |

The rate correlation is not merely weak, it has **the wrong sign** — higher reward
should mean *better* avoidance and therefore a *lower* rate. And the two readouts
only agree with each other at ρ = −0.31, when they are supposed to be two views
of one quantity.

| scale | rate (sd) | margin Δ (sd) |
|---|---|---|
| 0.02 | 0.078 (0.10) | 1.89 (1.06) |
| 0.05 | 0.046 (0.05) | 2.13 (1.71) |
| 0.15 | 0.117 (0.12) | 3.76 (1.94) |
| 0.40 | 0.036 (0.04) | 2.33 (0.93) |
| 1.00 | 0.126 (0.06) | 1.38 (1.28) |

Neither column is monotone. Within-rung standard deviations are as large as the
between-rung differences. **16 of 20 organisms learned**, at every scale from 0.02
to 1.0 alike.

## The organism has two states, not a range

This is pre-commitment (3) firing:

> *"If the margin ALSO saturates, then the organism genuinely has only two states
> and the dose-response design cannot be run on this task at all — which would be
> a significant negative result about the task, not about the instruments."*

The margin does not saturate — it varies from 1.38 to 3.76 — but it does not vary
**with dose** (ρ = 0.00). The variance is seed-to-seed, not dose-to-dose. So the
conclusion stands in its stronger form: **on this task the organism either finds
the avoidance rule or does not, and reward magnitude across two orders of
magnitude does not modulate how strongly it holds it.**

## My pass criterion was wrong, and it passed

The run's own recorded verdict reads:

> `RATE AXIS USABLE -- reward_scale grades the organism; proceed with rate as
> realised dose and margin as the finer readout.`

**That verdict is false and the fault is in the criterion I pre-registered**, which
asked:

```python
rate_usable = rate_spread["sd"] >= 0.05 and rate_spread["max"] > 0.0
```

That is a test for *"is there spread?"*. Seed-to-seed noise satisfies it exactly as
well as a real dose effect does. The question a dose axis actually poses is
whether the readout **tracks the dose**, which is a correlation, not a variance.
I wrote a pre-commitment that could not fail for the right reason.

The criterion is now corrected in `run.py` to a rank correlation against
`reward_scale`, with the old behaviour and this history recorded in the code
rather than quietly swapped. The stored JSON keeps the original verdict string; it
should be read as an artefact of the bug, not as a finding.

Worth stating plainly: **pre-registration protects against choosing an
interpretation after seeing the data. It does not protect against pre-registering
a bad test.** This one only surfaced because the per-rung means were printed and
were visibly non-monotone.

## What this costs the program

The design's claim to be a **measurement** study rather than a **detection** study
rested on dose-response: vary function, watch each instrument's reading move, read
off a slope. With no usable dose knob:

- The loading map degrades to a **binary contrast** — organism vs control — which
  is the 2×2 the continuous design was explicitly built to improve on.
- **H4 (non-additivity at the centre point) loses its only foothold.** ORG-C's
  additivity check needs graded arms to compare against.
- Instrument *loadings* become instrument *discriminations*: "can this readout
  tell a trained organism from an untrained one", not "how steeply does it
  respond".

This does **not** invalidate the screening claim, which only ever needed ORG-A to
differ from ORG-B. It removes the upgrade from screening to measurement.

## What might still recover a dose axis

None of these are free, and none should be attempted before E12/E13 report:

1. **Training steps as the knob**, which E11 explicitly rejected as a
   procedure-property rather than an organism-property. That objection stands, but
   a step ladder at least *has* a monotone relationship to how much learning
   happened, which reward magnitude turns out not to have.
2. **Partial training with early stopping on the margin** — pick organisms off a
   single run at graded margin values. Cheap, and it makes the margin the dose by
   construction rather than by hope.
3. **Harder tasks** — a grid where perfect avoidance is impossible would put a
   ceiling below saturation and let magnitude matter.
4. **Accepting the binary axis** and reporting a detection study honestly.

Option 2 is the cheapest and the most likely to work, because it stops trying to
control the outcome through an input the optimiser evidently normalises away.

## Threats

- 4 seeds per rung. Within-rung sd is large, so a *weak* dose effect could hide
  here. What is ruled out is an effect large enough to build a design on.
- One task, one model, one RL configuration.
- The margin is measured greedily at the first token on held-out states; a
  different readout could in principle grade where this one does not.

## Next

1. E12 (instruments) and E13 (organism set) proceed unchanged — both need a
   trained-vs-untrained contrast, which exists, rather than a graded one.
2. Do not build the loading map as a slope. Build it as a discrimination.
3. If a dose axis is wanted later, take option 2 above.
