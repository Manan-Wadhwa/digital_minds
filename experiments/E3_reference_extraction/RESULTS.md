# E3 — the reference extraction, implemented. My hypothesis was wrong.

**git** `f1c9d02` · 2026-07-30 · Qwen3-4B-Instruct-2507, **untrained**
2,700 trajectories (434 penalised / 356 rewarded / 1,910 path), lengths 1–15,
3 counterbalanced seeds · action-token readout · ~60 s

---

## Headline

**Switching to the reference's readout did not reproduce their number, and did not
even move ours.**

| | cos(v_pen, v_rew) |
|---|---|
| reference, pre-training | **−0.23 … −0.13** |
| ours, prompt-token adjacency (E1a) | **+0.79** |
| ours, **action-token, landing classes** (E3) | **+0.83** |

Range across all 36 layers: **+0.706 … +0.888**. Layers falling inside the
reference band: **0 / 36**.

## The claim this falsifies — mine

When I wrote `reference_extraction` I put this in its docstring:

> *"At the action token over landing classes the shared term is not 'a tile is
> visible' — both classes are equally visible — but 'I committed to a move', which
> is common to all three classes including the Path baseline and therefore
> subtracts out."*

**That is wrong, and the reasoning was sloppy in a way I should have caught.** Path
is still the baseline, so both vectors are still *"landed on a non-Path tile"*
minus *"landed on Path"*. The shared term isn't "a tile is visible" and isn't "I
committed to a move" — it is **"non-Path"**, and it has exactly the same
structure as the salience component that produced E1a's +0.80. I changed the
readout site and the sample unit while leaving the thing that actually caused the
problem untouched.

Fifth time a measurement has overturned a confident claim in this program. The
docstring was written in the same commit as the code, before any data.

## What was actually learned, and it is more useful

**Landing-class separability on the untrained model is 0.507–0.608 against a
chance of 0.500.** Barely above chance, and *nothing like* E1b's 0.98 for
prompt-token tile identity.

That contrast is the real finding:

| quantity | untrained separability |
|---|---|
| which tile is **visible** (prompt token) | **0.98** — saturated |
| which tile the model **lands on** (action token) | **0.61** — near chance |

The model encodes what it can see almost perfectly, and barely encodes the
*consequence of its own action* — which makes sense, because an untrained model
is not trying to reach anywhere. Landing somewhere is not yet a thing it is doing.

## This hands us the gate we have been failing to find

Three candidates have now failed:

| candidate | why it failed |
|---|---|
| cosine | baseline-dependent (+0.79 / −0.07 / −1.00), and now robustly positive under both readouts |
| prompt-token probe | **saturated** at 0.98 untrained |
| tile↔move alignment | usable floor 0.05, but no external anchor |

**Landing-class separability has neither defect:**
- **Not saturated** — 0.61, with 0.39 of headroom.
- **No baseline to choose** — it is a classification accuracy, not a contrast.
- **It measures the right thing.** RL is supposed to make the model care about
  where its move lands. If "recruitment" means anything behaviourally, the
  representation of the landing tile should strengthen. That is a directional
  prediction, made before training, on a quantity that starts near chance.

## Threats

- **Their number is still unreproduced**, and I cannot say why. Either the spec has
  an element I have not recovered from the prose, or their maze geometry, tile
  densities, episode structure or model differ materially. **Do not claim a
  reproduction anywhere.**
- The positive cosine is now robust across two very different readouts and sample
  units, which suggests it is a property of *this* setup rather than a measurement
  artefact — a fact that argues against my "wrong readout" story too.
- Class sizes are unbalanced (1,910 path vs 356 rewarded); separability is a mean
  over pairwise probes, so it is affected. Balance before quoting.
- 2,700 trajectories vs the reference's 5,000 *per class*. Under-sampled.

## Next

1. **Adopt landing-class separability as the gate.** Pre-register the floor
   (0.61) and the prediction (rises with RL) before any organism is trained.
2. Balance the classes and re-measure the floor with seeds.
3. Fix the entropy collapse (E2a), then train ORG-A and test the prediction.
