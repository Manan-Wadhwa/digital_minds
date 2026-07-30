# E6 — the landing-class gate loses to a bag of token counts. And E5's numbers now have a floor.

**git** `7c8427f` · 2026-07-30 · Qwen3-4B, **untrained** · 2 counterbalance
families × 4 independent rollouts · 3,000 trajectories each · no training

---

## Headline

Two results, and the second is fatal to the gate.

**1. The noise floor is SD = 0.0147.** Eight independent rollouts from the *same
frozen model*, glyph roles pinned so only sampling varies. Every difference E4 and
E5 reported can finally be read against something.

**2. A bag-of-token-ids classifier scores 0.660. The activation probe scores
0.626.** The surface baseline **beats** the probe by +0.034 — more than two noise
SDs, in the wrong direction.

`surface_baseline`'s own docstring, written for E1a and never pointed at this gate:

> *"If an activation probe scores below this, the probe is not reading a richer
> representation — it is reading a worse copy of the input."*

That is what happened. **The landing-class gate is dead**, and not for any reason
that better organisms, longer training or a cleaner protocol would fix. The tile a
step lands on is a deterministic function of the grid and the emitted letter, both
verbatim in the probed text, and token counts recover it *better* than the residual
stream does.

## E5's findings, rescored against the floor

| E5 effect | value | in SD units | verdict |
|---|---|---|---|
| composition, raw | +0.0375 | **2.55 σ** | **survives** |
| composition, balanced | +0.0761 | **5.18 σ** | **survives strongly** |
| representation, raw | −0.0007 | −0.05 σ | zero, confirmed |
| representation, balanced | +0.0163 | 1.11 σ | **inside noise** |

E5's central claim holds: **the separability rise was composition drift, and that
conclusion is now backed by a measured floor rather than an assumption.**

And the one hopeful sign from E5 dies here. E5 reported that its single learner had
the largest balanced representation effect (+0.065 vs +0.007), flagged as n=1 and a
hypothesis. **The whole balanced representation term is 1.11 σ — indistinguishable
from noise.** There was nothing there to be hopeful about.

## The measurements

| family | fixed-layer null | balanced | surface | split-half |
|---|---|---|---|---|
| 0 (penalised = blue) | 0.625 ± 0.010 | — | 0.651 ± 0.035 | 0.591 ± 0.015 |
| 1 (penalised = purple) | 0.627 ± 0.020 | — | 0.669 ± 0.027 | 0.600 ± 0.021 |
| **pooled** | **0.626 ± 0.0147** | | **0.660 ± 0.0307** | 0.595 ± 0.018 |

Per repeat:

| family | rep | layer | sep | balanced | **surface** | counts (pen/rew/path) |
|---|---|---|---|---|---|---|
| 0 | 0 | 11 | 0.633 | 0.595 | 0.617 | 545/340/2115 |
| 0 | 1 | 14 | 0.638 | 0.610 | **0.657** | 517/339/2144 |
| 0 | 2 | 18 | 0.648 | 0.631 | 0.631 | 546/340/2114 |
| 0 | 3 | 16 | 0.633 | 0.636 | **0.697** | 497/326/2177 |
| 1 | 0 | 13 | 0.656 | 0.634 | 0.634 | 370/536/2094 |
| 1 | 1 | 27 | 0.623 | 0.639 | **0.699** | 342/540/2118 |
| 1 | 2 | 29 | 0.637 | 0.652 | **0.668** | 369/536/2095 |
| 1 | 3 | 19 | 0.638 | 0.668 | **0.677** | 335/535/2130 |

Surface beats the activation probe in **6 of 8** repeats.

## Pre-commitments, scored

| # | prediction | outcome |
|---|---|---|
| 1 | null SD 0.010–0.020, max gap < 0.05 | **hit** — SD 0.0147, range 0.046 |
| 2 | surface high, ≥ 0.85; if surface > activation the gate is dead | **half hit, and the half I got wrong matters less than the half I got right** |
| 3 | split-half spread larger than between-rollout spread | **marginal** — 0.018 vs 0.015, same order |

On (2): I predicted the surface baseline at **0.85+** and it came in at **0.660**.
My magnitude was badly wrong — a bag of counts discards tile *positions*, which is
most of what determines a landing tile, so it should never have been expected near
ceiling. But the disqualifying comparison was never against 0.85; it was against
the activation probe, and that comparison came out exactly as pre-registered.

The argmax layer wanders across repeats (11, 14, 18, 16 / 13, 27, 29, 19) with no
stability whatsoever. That is worth stating on its own: **"the layer of maximum
separability" is not a well-defined quantity here**, which retroactively undermines
using it as a selection criterion — as E3 proposed and E4/E5 both did. The
fixed-layer spread is reported precisely so this wandering does not inflate the floor.

## Why this happened, and it is the same reason twice

E5 killed the gate's *measurement*; E6 kills the gate's *quantity*. Both failures
have one root cause:

> **The thing being probed was a property of the input, and the input was allowed
> to carry it.**

Composition let the input set move between conditions. Surface shows the label was
recoverable from the input all along. Any gate of the form *"how well can I decode
X from activations"* inherits both, because X was in the text before the model did
anything.

This is now four gate candidates and three distinct failure modes:

| candidate | failure |
|---|---|
| cosine, prompt-token | baseline-dependent: +0.79 / −0.07 / −1.00 on identical activations |
| probe separability, prompt-token | saturated at 0.98 untrained |
| tile↔move alignment | usable floor, no external anchor |
| **landing-class separability** | **composition drift (E5) + loses to token counts (E6)** |

## Threats

- The surface model is bag-of-**counts**, so it has no positional information and
  is a *weak* surface baseline. A stronger one (n-grams, positions) would only widen
  the gap. **This threat runs against the gate, not for it.**
- 8 repeats, one model, one grid size.
- The split-half arm barely separates probe-fitting noise from rollout noise. It is
  reported for completeness and no weight is placed on it.
- Surface SD (0.031) is twice the activation SD (0.015), so single-repeat surface
  numbers are noisier than single-repeat probe numbers. The 6-of-8 count and the
  pooled means are the reliable comparison, not any one row.

## Next

1. **Stop probing decodable properties of the input.** E8 is written against this:
   paired activations on *identical* trajectories, base vs trained weights, so
   composition and surface are both zero by construction and the measured quantity
   is selectivity of the weight-induced shift, against a permutation null.
2. E7 (yield) still matters — E8 needs learners — but it is no longer the only
   blocker, because the gate would have failed at any yield.
