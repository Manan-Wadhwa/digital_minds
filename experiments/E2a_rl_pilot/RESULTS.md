# E2a — first RL run: cost measured, organism did not learn

**git** `d5f8e1b` · 2026-07-30 · Qwen3-4B-Instruct-2507 + LoRA r=16 on `q_proj`/`v_proj`
500 steps × (16 states × 8 samples) · lr 1e-4 · temperature 1.0 · seed 0

---

## The number everything was waiting on

```
0.111 s/step   ·   500 steps = 56 s   ·   one organism ≈ 1 minute of GPU
```

**Planning consequence, and it is large.** The full design — 8 cells × 3 seeds,
plus ORG-A′ and ORG-B′ — is roughly 30 organisms, i.e. **~30 minutes of GPU
total.** Every compute-budget argument in the program has been arguing about
nothing. The binding constraint is engineering and analysis time, not the GPU.

This retires several open questions at once: seeds are essentially free (take 5,
not 3), a second model family is affordable (AB9 is back on the table), and the
scale question is not a resource trade-off.

## But the organism did not learn

| | penalised_rate |
|---|---|
| before | 0.188 |
| after 500 steps, held-out states | **0.211** |
| tile-density chance | 0.200 |
| uniform-random-move baseline | **0.209** |

It ends **at or slightly above** the random-move baseline. The manipulation check
fails: this is not ORG-A.

## Diagnosis: entropy collapse

```
step   0   H 0.952   |g| 41.3
step 100   H 0.001   |g| 0.000     <- collapsed
step 200   H 0.147   |g| 52.2      <- brief recovery
step 300   H 0.001   |g| 0.000
step 400   H 0.000   |g| 0.000
step 499   H 0.000   |g| 0.000
```

Mean reward oscillates with no trend: 0.86 → 9.33 → −0.70 → 4.33 → −0.08.

**The mechanism is structural to the single-step formulation.** With 4 actions and
a group of 8, once the policy sharpens every sample in a group is the *same*
action → reward is constant within the group → advantage `r − mean(r)` is
identically zero → no gradient. The policy freezes on whichever action it
collapsed to, which is not the good one, and cannot escape because escape
requires exploration it no longer has.

## Two things to fix before the next attempt

**1. The zero-signal guard did not fire.** `zero_signal_steps` reports **0** while
the log shows `|g| 0.000` at steps 100, 300, 400 and 499. The guard tests for an
exactly-zero gradient norm; the true value is presumably ~1e-8 rather than 0.
Needs a tolerance, not an equality test. Worth noting the guard was added by its
author after finding that Adam's momentum walks a *converged* policy off its
optimum when advantages vanish — so this is the same failure it was written for,
still live.

**2. Entropy collapse needs an actual remedy**, not just detection. Candidates, in
order of how much I'd bet on them:
- **Entropy bonus** `−β·H` in the loss. Directly targets the mechanism.
- **Lower lr.** 1e-4 with Adam on 5.9M adapter params reaches determinism in
  ~100 steps; 1e-5 is the obvious first try.
- **Sampling temperature > 1.** Cheapest, but treats the symptom.
- **Larger group.** Does not help — collapse makes all G identical regardless.

## What this does not mean

Nothing here says the maze is unlearnable or that Dr.GRPO is wrong. It says the
*hyperparameters* put the policy into a degenerate corner within 100 steps. At
0.111 s/step a sweep over lr × entropy-bonus × temperature costs minutes, so this
is cheap to resolve — the point of running a pilot.

## Verified along the way

- LoRA is exactly the identity at step 0: max |logit change| **0.000e+00**.
- Gradients reach LoRA params only: **0** grad-enabled params outside adapters.
- `remove_lora` restores the base model exactly: **0.000e+00** change.
- 5.90M trainable / 4.03B total = **0.146%**.
- Untrained policy sits at chance (0.188 vs 0.200/0.209) — the pre-training
  manipulation check passes, confirming E1d's neutrality result behaviourally.

## Next

1. Fix the zero-gradient guard tolerance.
2. lr × entropy-bonus sweep. Minutes of GPU.
3. Only then treat any run as ORG-A.
