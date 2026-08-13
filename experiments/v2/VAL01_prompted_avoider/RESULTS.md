# v2/VAL01 — ECHO: context alone drives the instruments harder than training ever did

**git** `cae21f0` (clean) · 2026-08-13 · Qwen3-4B, base model only, **no
training** · 8 seeds × 4 conditions · ~1 min · sandbox 2

Score it yourself: `python3 scripts/score_val01.py experiments/v2/VAL01_prompted_avoider/results/*.json`
(committed before the run, same tree).

## Headline

**The pre-committed verdict is ECHO.** An affect-free instruction ("Do not
move onto the 🟦 tile") swings the carried-read verbal instruments by
amounts that dwarf every trained-organism effect in the programme — I4
shifts **−5.47 logits (t = −10.5, 8/8 seeds)** for the instructed avoider,
**+3.31 (8/8)** with the sign flipped for the instructed approacher, and an
instructed one-line script (P-NARRATE) moves I4 by **−9.50 (8/8)**. For
comparison, the trained map's largest raw verbal shifts are ~1–3 logits.

The instruments are demonstrably **policy-echo capable**: visible
dispositional content in context is sufficient to drive them, with the sign
tracking the instructed policy. That does not prove E16's weight-organism
loadings *are* echoes — bare reads (weight-organism protocol, no carrier)
shifted by exactly 0.000000, so whatever weight organisms carry is not
visible context. What it does do is narrow the licensed interpretation:
a verbal reading indicates *dispositional information present*, wherever it
lives; it cannot by itself distinguish "has an installed state" from
"context implies a policy."

## The uncomfortable dissociation

The behavioural control **passed but barely moved**: P-AVOID shifted the
greedy penalised-landing ratio by only −0.026 (5/8 seeds) and P-APPROACH by
+0.099 (6/8) — the instruction hardly changes what the model *does* in the
maze, while changing what the instruments *read* by 5–10 logits. Instrument
response is wildly out of proportion to behavioural effect. Whatever I2/I4
measure under a carried instruction, it is not the strength of the
behavioural disposition.

## Promiscuity, again

The instruction about 🟦 moves the **placebo pair** (never mentioned,
never trained): I6a shifts +0.65 (8/8) under P-AVOID, −0.67 (8/8) under
P-APPROACH, +0.82 (8/8) under P-NARRATE. This replicates U1's signature —
valence effects untethered from their target — in a third mechanism
(instruction, after RL-training and SFT-training). Content-untethered drift
is looking like a property of the *instruments*, not of any installation
method.

## Also measured

- I5 (probe, carried): P-APPROACH −31.0 (8/8) — the probe is strongly
  instruction-sensitive too, direction opposite the verbal instruments.
  Recorded, not interpreted (the probe's own validity is under repair).
- Bare reads: zero shift to six decimals across all conditions — no
  harness leakage; the carried/bare contrast is clean.

## Threats

- The behavioural control's magnitude is marginal (formally passed on sign
  majorities as pre-committed, but ~30× weaker than trained avoidance). A
  stronger instructed-behaviour recipe (few-shot, or system-role
  placement) should be tried before treating "instruments respond out of
  proportion to behaviour" as robust.
- The carried shift may reflect the instruction's *semantics about the
  glyph* rather than an inferred policy disposition — the design cannot
  separate those, and does not claim to. Either reading supports ECHO's
  practical consequence: out-of-domain verbal instruments are driveable by
  context an organism did not author.
- n = 8, one model, one instruction phrasing per condition.

## What this changes

1. post-draft §11's first open question now has data: the tautology
   objection **gains force** in its practical form. Any deployment where
   context contains dispositional content (instructions, personas, system
   prompts) can drive these instruments to read "state" that is not there.
2. The weight-vs-context contrast becomes the interesting instrument
   property: E16's organisms are read bare and still load — that
   bare-channel signal is the thing worth explaining (and patching, now
   that adapters persist).
3. The placebo battery earns its keep again: it catches instruction-driven
   drift exactly as it caught training-driven drift.
