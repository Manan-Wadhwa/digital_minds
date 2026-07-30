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

## v3: the fix, and what it predicts

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

- **ORG-A does not reproduce E9** (0.98 / 0.23 / 0.15 / 0.87 against E9's 0.086)
  even after matching E9's seeding. Something else differs in E13's RL path and it
  is unresolved. ORG-A′ is currently the more dependable functional organism.
- 4 seeds per kind. Every "n/4" here is a small number.
- The narration check greedily decodes and string-matches the aversive list, so a
  paraphrasing organism scores low despite narrating. Generations are saved
  verbatim for audit.
- ORG-C has one seed at 1.057 against two at ~0.037 — bimodal, not merely noisy.

## Next

1. v3 with the anchor; compare against v2 on the same seeds.
2. If B holds, re-run the instrument battery (E12) against a valid set — its
   current loadings were computed against the pre-fix organisms and are stale.
3. Resolve ORG-A vs E9.
