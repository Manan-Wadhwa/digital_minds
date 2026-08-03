# E13 — the organism set, and why the two axes fought each other

**git** v1 `c40bbb5` · v2 `1f3d9c8` · v3 `7e49f0d` · 2026-07-30 ·
Qwen3-4B + LoRA r=16 · 6 kinds × 4 seeds · ~30 min per build

---

## What this had to produce

Organisms whose internal structure is known by construction, because the whole
program is a comparison against them:

| kind | function | narration | method |
|---|---|---|---|
| ORG-A | ✓ | ✗ | RL avoidance |
| ORG-A′ | ✓ | ✗ | SFT on oracle moves — **method-matched to B** |
| ORG-B | ✗ | ✓ | SFT: base-policy moves + aversive remark |
| ORG-B′ | ✗ | ✗ | SFT: base-policy moves + affectless remark |
| ORG-C | ✓ | ✓ | RL, then aversive commentary |
| ORG-D | ✗ | ✗ | base model |

`ratio` = penalised-landing rate ÷ random-move rate. Functional organisms must
sit **below 0.75**; narration organisms must stay **within ±15% of 1.0** — a
two-sided band, because a policy that moved either way has moved.

## v1 → v2: both bugs fixed, and the fix for one broke the other

| kind | v1 (384 SFT ex.) | v2 (1536 SFT ex.) | |
|---|---|---|---|
| ORG-D | 1.043 · 4/4 | 0.94 1.11 1.05 1.07 · **4/4** | ✅ |
| ORG-A | 0.557 · 2/4 | 0.98 0.23 0.15 0.87 · 2/4 | ⚠️ |
| **ORG-A′** | 0.889 · **0/4** ❌ | **0.26 0.34 0.22 0.63 · 4/4** | ✅ **fixed** |
| **ORG-B** | 1.072 · **3/4** ✅ | 0.94 **1.18 1.20** 1.03 · **2/4** | ❌ **broken** |
| ORG-B′ | 1.041 · 4/4 | 0.98 **1.14 1.16** 1.07 · 3/4 | ⚠️ drifting |
| **ORG-C** | 1.052 · **0/4** ❌ | 1.06 **0.038 0.036** · 2/3 | ✅ **fixed** |

### ORG-C was a bug, not a limitation

Its commentary SFT used **base-policy moves**, so stage two was training it to
move like the *untrained* model while narrating — erasing the RL avoidance it had
acquired moments earlier (0.229 → 1.052). Fixed by giving it oracle moves. It now
reaches 0.036–0.038, near-perfect avoidance *with* narration.

**Worth recording how close this came to being published as a finding.** My
pre-commitment read: *"I expect ORG-C's function check to be weaker than ORG-A's…
a C whose function has decayed is a known limitation to report, not a result
about interaction."* That framing was ready to absorb a plain bug as interference
between training stages, and the number it predicted was the number the bug
produced. Pre-registering the right prediction for the wrong reason is worse than
not predicting, because it makes the bug feel expected.

### ORG-A′ was data volume

384 grids with one label each, against RL's ~51,000 sampled actions. At 1536 it
becomes **the most reliable functional organism in the set — 4/4, beating ORG-A's
2/4.** Direct supervision does win, but only at comparable volume.

### And that same increase broke ORG-B

`sft_examples` was one shared knob and the two axes wanted opposite values.

## The mechanism, and why it was diagnosable

ORG-B's completion is `"move. remark"` and the loss covered **both**, with the
move target a hard **sample** from the base policy. Repeatedly fitting samples
drawn from a distribution sharpens the policy toward whichever samples were
drawn — self-distillation, with an effect that grows with volume.

That predicts the measured pattern exactly: **1.072 at 384 → 1.18–1.20 at 1536.**

The confirming evidence is the control. **ORG-B′ drifts in lockstep with ORG-B**,
failing the same seeds (1 and 2) at nearly the same magnitude (1.14/1.16 vs
1.18/1.20). Two organisms with *opposite remark content* drifting together means
the cause is the move loss, not the commentary. Without B′ this would have looked
like "aversive training makes the model approach the aversive tile", which is a
far more interesting and completely wrong conclusion.

## v3: the anchor's effect is not measurable at this n

Ran with the anchor on, same seeds. **ORG-B went 2/4 → 4/4.** That looks like the
fix working. It is not readable as one.

| kind | anchor OFF (v2) | anchor ON (v3) | mean \|ratio−1\| |
|---|---|---|---|
| ORG-B | 0.943 1.181 1.200 1.030 · 2/4 | 0.906 1.143 0.982 1.069 · **4/4** | 0.117 → **0.081** |
| ORG-B′ | 0.981 1.143 1.164 1.069 · 3/4 | 0.868 **1.333** 0.982 1.109 · 3/4 | 0.099 → **0.148** |

**The two anchored organisms moved in opposite directions.** B improved by 0.036;
B′ got worse by 0.049. Both are anchored, so a real anchor effect should move them
the same way.

### The noise floor, measured rather than assumed

> ❌ **CORRECTED 2026-08-02 — this is not a noise floor.** The table's "changed
> between runs? no" column is wrong for every row. Commit `7e49f0d`, which is
> what separates v2 from v3, changed the per-kind reseed from
> `set_all_seeds(seed + 1)` to `set_all_seeds(seed)`, and `inject_lora` draws its
> A matrices from the global RNG on the next line. **Every organism in v3 started
> from a different adapter initialisation than in v2.** B is zero-init so the
> model is identical at step 0, but `dL/dB ∝ A·x`, so A shapes the first update
> and everything after it.
>
> So 0.104 and 0.293 measure **initialisation sensitivity**, not run-to-run
> variance. They remain real and useful — that is how far an organism moves when
> its adapter init reseeds — but they do not bound a comparison in which the init
> was held fixed, and they are not evidence about GPU nondeterminism.
>
> The direct measurement points the other way: ORG-A reproduces **bit-identically**
> across E13 v3 and E14 (`0.4906 0.7619 0.2545 0.9109`), two runs half an hour
> apart in different experiment files, after 800 RL steps each. See HANDOFF §2c.
>
> The anchor conclusion below is unaffected in direction — 0.036 is still small
> against 0.104 — but "the noise it would have to clear" is the wrong description
> of what 0.104 is. The anchor test compared two runs that differed in the anchor
> **and** in every organism's initialisation, so it remains unable to isolate the
> anchor. Both conditions inside ONE run, as §"What this means" says.

| organism | training | changed between runs? | mean per-seed movement |
|---|---|---|---|
| ORG-D | none | **init reseeded, but untrained** | **0.000** (bit-identical) |
| ORG-A′ | SFT | **yes — adapter init** | **0.104** |
| ORG-A | RL, 800 steps | **yes — adapter init** | **0.293** |

```
anchor effect on ORG-B  0.036
SFT noise floor (A')    0.104
effect / noise          0.35      must exceed 1 to be readable
```

**The anchor's apparent effect is about one third of the noise it would have to
clear.** ORG-B reaching 4/4 is consistent with the anchor working and equally
consistent with a lucky draw. ORG-B′ moving the other way is the tell.

### Why the noise exists

> ❌ **RETRACTED 2026-08-02.** This section said training is nondeterministic and
> evaluation is not. Training here is deterministic too: ORG-A is bit-identical
> across E13 v3 and E14. What follows was written from a comparison whose code
> **had** changed — see the correction above and HANDOFF §2c — and the phrase
> "with no code change" is simply false; the change is in `7e49f0d`.
>
> The corrected reading of the same two numbers: RL (0.293) is more sensitive to
> its initialisation than SFT (0.104), which is still what you would expect from
> 800 sequential on-policy updates versus 768 supervised ones, and is still worth
> knowing. ORG-D is 0.000 because it is never trained, so its init is irrelevant
> rather than because it is the only deterministic thing in the run.

~~ORG-D returns **bit-identical** ratios across runs; ORG-A, with no code change,
moves by 0.29. Evaluation is deterministic and **training is not** — GPU
reduction order is nondeterministic and the divergence compounds over hundreds of
gradient steps. RL (0.293) is far noisier than SFT (0.104), as expected from 800
sequential updates versus 768.~~

### What this means

- **The anchor is neither confirmed nor refuted.** Its construction is still
  correct — four tests pin that it is zero at init, monotone in drift, and pulls
  back — and its final value is small but non-zero (B 0.040, B′ 0.014), so it is
  doing something. Whether that something preserves the policy is unmeasured.
- **Deciding it needs both conditions inside ONE run**, anchored and unanchored
  organisms trained back-to-back from the same base, or roughly 10× the seeds.
  A v2-vs-v3 comparison cannot do it and I should not have designed it that way.
- **This retroactively weakens per-seed claims made earlier in this program.**
  Any statement of the form "fix X moved seed N from a to b" across two runs is
  suspect at this effect size.
  *(Still true, corrected reason — 2026-08-02. Not because two runs of the same
  code disagree; they do not. Because the runs being compared silently differed
  in more than the fix: the adapter init reseed in `7e49f0d`, and the shared-`gen`
  draw order that redrew 66% of ORG-A′'s labels between E13 and E14. With
  `derive_generator` and a pinned init, a two-run comparison is legitimate again —
  the requirement is to verify the inputs were pinned, not to assume they cannot
  be.)*

I am reporting ORG-B as **2/4 with a plausible but unproven fix**, not as 4/4.
The set is usable for E14 either way — the loading map contrasts *groups* of
organisms and the group means are stable — but ORG-B's policy invariance is a
stated limitation, not a settled property.

## v3's original prediction, scored

**Predicted:** seeds 1 and 2 return inside the band for both B and B′, A′ and C
unchanged. **Outcome:** seeds 1 and 2 did return inside the band for B, but B′
seed 1 moved *further out* (1.143 → 1.333), and the whole comparison sits under
the noise floor. Scored as **not established**.

## The original v3 rationale (retained)

Anchor narration organisms to the base policy's **distribution** rather than a
sample of it — `KL(base ‖ current)` on the move token:

- at step 0 the model **is** the base policy, so the term is **exactly zero** and
  contributes no gradient;
- as remark training perturbs the move logits it becomes a restoring force.

A leash rather than a push. Volume stays matched between A′ and B, so the
method-matching that A′ exists to provide survives — which the cheap fix
(per-kind `sft_examples`) would have destroyed.

**Pre-registered prediction:** seeds 1 and 2 return inside the ±15% band for both
B and B′, while ORG-A′ and ORG-C are unchanged (they are not anchored, and are
*supposed* to move their policy).

**Pre-registered failure reading:** the anchor removes *direct* pressure on the
move distribution, but remark training still updates shared weights. If B still
drifts with the anchor on, the residual is indirect and the honest conclusion is
that **narration cannot be installed by fine-tuning the same weights that carry
the policy** — it needs a separate head, or prompt-side narration.

Four tests pin the anchor's defining property before any GPU time: ~0 when the
policy is unchanged, monotone in drift, and one gradient step on it reduces the
divergence.

## Threats

- **ORG-A vs E9 was my misreading, now corrected.** I wrote this up as a
  reproducibility failure. It is not: **E9 tuned on held-out seeds 10–13 and E13
  runs the reporting seeds 0–3**, so there was never a number to reproduce.

  What the gap does show is a **transfer gap** — E9's configuration reaches 4/4 on
  the seeds it was selected on and about 2/4 on unseen ones. That is the honest
  and more useful reading, and it is exactly what held-out tuning is for. E9's
  0.086 must be quoted as "on the tuning seeds", never as ORG-A's expected value.

  ORG-A′ is nonetheless the more dependable functional organism (4/4 vs 2/4).
- 4 seeds per kind. Every "n/4" here is a small number.
- The narration check greedily decodes and string-matches the aversive list, so a
  paraphrasing organism scores low despite narrating. Generations are saved
  verbatim for audit.
- ORG-C has one seed at 1.057 against two at ~0.037 — bimodal, not merely noisy.

## Next

1. v3 with the anchor; compare against v2 on the same seeds.
2. **E14 — the loading map.** Every instrument gets a function loading and a
   narration loading against the organism kinds. This is the deliverable and it
   is queued behind v3.
3. E12's loadings are stale (computed against pre-fix organisms) and are
   superseded by E14 rather than re-run.
