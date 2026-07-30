# E7 — 0/24. A fixed entropy bonus cannot hold the policy open at 800 steps.

**git** `5b963ba` · 2026-07-30 · Qwen3-4B + LoRA r=16 · Dr.GRPO 800 steps ·
6 configurations × 4 **held-out** seeds (10–13) · reporting seeds 0–5 untouched ·
0.079 s/step

---

## Headline

**Not one organism learned, in any cell.**

| lr | entropy_coef | learned | mean ratio | worst | zero-signal steps | mean final entropy |
|---|---|---|---|---|---|---|
| 1e-4 | 0.003 | **0/4** | 0.895 | 0.991 | 1717 | 0.000 |
| 1e-4 | 0.01 | **0/4** | 1.042 | 1.219 | 206 | 0.000 |
| 1e-4 | 0.03 | **0/4** | 1.000 | 1.219 | 167 | 0.001 |
| 3e-4 | 0.003 | **0/4** | 0.970 | 1.067 | 2261 | 0.000 |
| 3e-4 | 0.01 | **0/4** | 0.932 | 1.062 | 1085 | 0.000 |
| 3e-4 | 0.03 | **0/4** | 0.856 | 1.062 | 167 | 0.000 |

`ratio` is held-out penalised rate ÷ random-move rate; the bar for "learned" is
< 0.75. Several cells sit **above 1.0** — worse than random.

**Final policy entropy is 0.000 in 23 of 24 runs** (the 24th is 0.001), against a
maximum of ln(4) = 1.386. Every single run collapsed to a deterministic action.

## What this actually shows, which is not what I predicted

I pre-registered in this experiment's own docstring:

> *"Steps is the lever I expect to matter most. E4 and E5 both show rates drifting
> down slowly rather than collapsing, which reads as under-training rather than
> instability. I expect 800 steps to beat 300 at matched lr and entropy_coef."*

**That was wrong, and backwards.** E4 and E5 at 300 steps produced 2/6 and 1/6.
The same configurations at 800 steps produce 0/4. 300 steps was not too few —
**800 is long enough to reach a collapse that 300 stopped short of.**

And the collapse is structural, not a bad hyperparameter. A fixed entropy bonus is
a *constant* pressure against a reward gradient that **grows as the policy
sharpens**. For any constant there is a horizon past which the policy goes
deterministic. Once it does:

- every sample in a group is the same action,
- reward is constant within the group,
- the Dr.GRPO advantage is **identically zero**,
- and learning stops with the policy frozen wherever it landed.

`zero_signal_steps` records exactly that, and it totals **5,603 dead steps out of
19,200** across the sweep. The 0.003 cells are worst (1717, 2261) because the
weakest bonus collapses earliest and then sits dead for hundreds of steps.

Raising the coefficient does not fix it, it only moves the horizon: `ec=0.03`
reduces zero-signal steps to 167 but still lands at entropy 0.000 and 0/4.

## Pre-commitments, scored

| # | prediction | outcome |
|---|---|---|
| 1 | 800 steps beats 300 (under-training) | **WRONG, and inverted** — 800 collapses where 300 merely underperformed |
| 2 | winner near ec 0.01, lr 1e-4 or 3e-4 | **no winner exists** |
| 3 | ec 0.03 may pin the policy near uniform; ec 0.003 at lr 3e-4 may collapse fast | **half right** — 0.003 collapsed as predicted (2261 dead steps), but 0.03 *also* collapsed to 0.000 rather than staying uniform. The degeneracy I named for one end happened at both. |
| 4 | if no cell reaches 3/4, stop tuning | **fired** |

## The verdict, and where I am departing from it

The run's own recorded verdict:

> `STOP TUNING -- yield is not a hyperparameter problem. Next moves, in order:
> more LoRA target modules (o_proj + MLP), larger batch_size, then reward shaping.`

**I am following the decision and not the list.** The decision — stop tuning a
fixed coefficient — is correct and is what the data says. But that list of next
steps was written before the run, and it addresses **insufficient capacity and a
noisy gradient**. Neither is the failure here. Adding adapter capacity to a policy
that has collapsed to a single action does not un-collapse it; it gives a dead
policy more parameters with which to stay dead.

The failure is that a constant bonus is outgrown. The fix that matches it is to
**target the entropy instead of fixing the coefficient**, so the pressure rises
exactly when the policy starts to sharpen too far:

    log_coef <- log_coef + entropy_lr * (target - measured_entropy)

That is implemented in `rl.py` (opt-in via `entropy_target`; default `None` leaves
every earlier experiment bit-identical) and tested in **E9**.

Deliberately **not** a KL penalty. `rl.py`'s header argues against a KL leash on
design grounds: the organism is *supposed* to depart from the base policy, so a
term pulling it back fights the manipulation check the run is scored on. An
entropy target constrains how *sharp* the policy may become, not *which* action it
prefers.

I am recording the departure rather than quietly substituting a plan, because the
whole point of pre-registering a fallback is that changing it should cost
something.

## What went right

- **Tuning was on held-out seeds 10–13 and reporting seeds 0–5 were never
  touched.** This is the error E4 made — selecting on seed 0, reporting on 0–5 —
  and it is now impossible here: `run()` asserts the tune set is disjoint from
  {0..5}.
- The sweep cost ~32 minutes and returned a clean structural answer rather than a
  marginal winner that would have been noise.

## Threats

- 800 steps only. It is possible some much longer schedule with a much larger
  coefficient works; nothing here rules that out, and nothing here recommends it.
- One task, one model, one formulation (single-step bandit).
- `evaluate_policy` uses 128 held-out states, so a ratio has meaningful sampling
  error; the finding rests on 24/24 agreement and the entropy collapse, not on any
  individual ratio.

## Next

1. **E9** — adaptive entropy targeting, on the same held-out seeds. Scores the
   mechanism (does entropy stay open?) **separately** from yield, because a
   controller that does not control makes yield uninterpretable.
2. If E9 holds entropy open and yield is still short, the honest move is the
   multi-step formulation `rl.py`'s header already anticipates — a trajectory
   return is a denser signal than one tile lookup — not further tuning.
