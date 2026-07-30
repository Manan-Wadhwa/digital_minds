# E1b — probe readout site, against a surface baseline

**Run** `20260730T150227Z_3640b9a024c2` · git `05b3136` · 2026-07-30
**Model** Qwen/Qwen3-4B-Instruct-2507, bf16, untrained · **Device** cuda:0
**Seeds** 5 (0–4), varying the state bank · **Wall clock** 28 s

![E1b readout comparison](results/e1b_readout.svg)

---

## Why this ran

E1a retired the cosine gate and nominated probe separability as its replacement,
then measured that probe at only **0.638** mid-stack. Two readings, which E1a
could not tell apart:

- **(a)** the model does not linearly encode adjacent-tile identity — the probe is
  a poor gate on the merits;
- **(b)** the readout site was wrong. E1a read the **final prompt token**, but the
  grid appears far earlier, so that position carries tile identity only insofar as
  attention has already moved it there.

## Headline

**(b), decisively. E1a understated the model by a wide margin — the defect was in
where we read, not in what the model represents.**

| readout | best layer | acc | @ L23 | vs surface |
|---|---|---|---|---|
| `last` (E1a's) | L1 | 0.752 | 0.586 | **below** |
| `mean_all` | L35 | 0.959 | 0.841 | above |
| `mean_grid` | L32 | **0.986** | 0.931 | **above** |

Surface baseline (bag of token ids on raw prompt text): **0.924** [0.879, 0.983].

## Findings

**1. The last-token readout is disqualified — by the pre-committed rule, not in
hindsight.** E1b's docstring stated before the run that an activation probe
scoring below the surface baseline "is not evidence of a representation; it is a
worse copy of the input." `last` reads 0.586–0.752 against a surface baseline of
0.924. It fails that test at every layer. **E1a's 0.638 was measuring attention
routing, not representation.**

**2. Grid-token pooling exceeds the surface baseline.** `mean_grid` reaches 0.986
at L32 and holds 0.931 at L23, against 0.924 for bag-of-tokens. A linear readout
of the residual stream over grid positions separates the conditions *better than
the raw text does* — the model is not merely preserving the input, it is
sharpening it.

**3. Probe separability survives as a gate — with a specification.** The gate is
usable only as *grid-pooled, mid-to-late layers*. Stated loosely as "probe
separability" it would have inherited exactly the kind of unstated-choice
sensitivity that killed the cosine gate. One readout choice moves the number from
0.586 to 0.931 on identical activations, which is the same failure mode as the
cosine's +0.79 / −0.07 / −1.00.

**4. The CV grid is misspecified, and fixed ridge beat it.** Cross-validated
accuracy at L23 for `mean_grid` is 0.931, but E1a's fixed `ridge=1.0` gives
**0.976** at the same layer. The selected alphas pin at the **lower bound** of the
`1e1…1e5` grid, so the optimum lies below 10 — outside the search. The CV numbers
above are therefore *conservative*, and the true separability is higher still.
This does not change any conclusion (every comparison is against the same
baseline) but the grid needs extending down before these numbers are quoted.

## What this changes

- **E1a's finding stands; its interpretation of the probe does not.** The cosine
  gate is still baseline-dependent and still retired. But "the probe is too weak
  to gate on" was wrong, and the U-shaped profile E1a reported was an artefact of
  last-token readout.
- **The replacement gate now has a specification:** grid-pooled probe
  separability, mid-to-late layers, reported against the surface baseline. Fix it
  in advance, like the estimator and pass rule in H1.
- **Every future activation measurement needs a surface baseline.** The one
  control that caught this was the trivial one. Without it, `last` at 0.752 would
  have read as a real if modest representation, when it is below what bag-of-words
  achieves.

## Threats to this result

- **Untrained model, adjacency contrast.** Still not the reference's extraction;
  the release question is still open and still the highest-value unblock.
- **`mean_grid` sees only grid tokens, which is where the manipulated glyph is.**
  That it beats surface is meaningful, but the comparison is not fully like-for-like
  — bag-of-tokens sees the whole prompt including boilerplate.
- **CV grid misspecified** (above). Absolute values conservative.
- Seeds vary the state bank only. Training-seed variance is untouched, because
  nothing has been trained yet.

## Next

1. Extend the CV alpha grid below 1.0 and re-run — cheap, 28 s.
2. Fix the gate specification in writing before any RL run.
3. First RL run, for the per-run cost that every scaling decision depends on.

## Reproduce

```bash
scripts/sync_to_sandbox.sh
# in kernel:  import run as E1b; E1b.run(model, tokenizer)
python3 scripts/plot_e1b.py \
  experiments/E1b_probe_readout/results/20260730T150227Z_3640b9a024c2.json \
  experiments/E1b_probe_readout/results/e1b_readout.svg
```
