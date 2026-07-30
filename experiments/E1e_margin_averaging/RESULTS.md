# E1e — margin averaging: failed by its own criteria, for a foreseeable reason

**Run** `20260730T155443Z_7262aa179e83` · git `c240d36` · 2026-07-30
**Model** Qwen3-4B-Instruct-2507, bf16, untrained · **Seeds** 6 · K ∈ 1..4 · **47 s**

---

## Verdict

**The fix failed against its pre-committed success criteria, and the mechanism
check failed too. Recording it as a negative rather than rescuing it.**

| pre-commitment | required | observed | |
|---|---|---|---|
| 0 · harness reproduces E1d | K=1 gap = E1d ±1e-3 | max diff **5.1e-06** | ✅ PASS |
| 2a · R² rises | ≥ +0.100 | **−0.093** | ❌ FAIL |
| 2a · seed spread contracts | SD ratio ≤ 0.667 | **1.111** | ❌ FAIL |
| 2c · noise falls like 1/K | share 0.91 → ~0.70 | **0.906 → 0.889** | ❌ FAIL |
| 3a · gap unmoved | move < 0.05 | 0.019 | ✅ pass |
| 3b · gate not inflated | max < 0.20 | 0.089 → 0.091 | ✅ pass |

## What happened

Mean best-layer R² **fell monotonically** with K: 0.453 → 0.392 → 0.374 → 0.360.

That monotone decline is the signature of the limitation the experiment
pre-registered but could not escape: **activations are captured under one order
while the target averages over K.** Every additional order moves the target
further from what the captured activations encode, so R² must fall with K whether
or not averaging removes noise. The design could not distinguish "averaging
doesn't help" from "the readout mismatch swamped the benefit."

**The lesson, which generalises beyond this experiment:** a pre-registered
limitation that can itself produce the null you are testing for makes the
experiment unable to fail informatively. The limitation was correctly stated in
advance and still cost a run — stating a confound is not the same as controlling
it.

## What was learned anyway

**Position noise is enormous.** Between-order SD of a single grid's margin is
**2.023** in logit units, and **~90% of target variance is list position, not
grid** (share 0.906 at K=1). The margin is a far worse target than E1c-2's
fixed-order R² of 0.68–0.83 made it look — that number was inflated because the
activations could read the fixed order and predict its contribution.

**The gate floor is robust to the analysis choice.** Seed-mean alignment
0.051 → 0.065, max 0.089 → 0.091. Well inside the 0.20 artefact threshold, and
close enough to E1d's floor that **E1d's `|cos| ≈ 0.05` stands.**

## Superseded by the reference specification

This whole line is now lower priority. The functional-welfare paper's extraction
turns out to be recoverable from the text (see `HANDOFF.md` §5), and it does not
use a prompt-token readout at all — it reads **the emitted direction token**,
conditioning on trajectories by which tile the model's own move lands on. That
sidesteps the list-position problem entirely, because the target is the model's
committed action rather than a margin computed over a shuffled option list.

**Do not buy K=16 before implementing the reference spec.** The correct next
experiment is theirs, not a bigger version of this one.

## If resumed anyway

Capture activations under all K orders and average those too, so target and
readout are matched. That is the only version of this experiment that can fail
informatively.

## Reproduce

```bash
scripts/sync_to_sandbox.sh experiments/E1e_margin_averaging/run.py
# in kernel: load by explicit file path (see HANDOFF §8), then E1e.run(model, tokenizer)
```
