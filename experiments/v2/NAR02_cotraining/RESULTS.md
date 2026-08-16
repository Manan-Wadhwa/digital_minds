# v2/NAR02 — the move-token objective is not the missing lever either, and the two arms that were supposed to move the policy collapsed it instead

**git** `ba576f47-dirty` · 2026-08-16 · Qwen3-4B-Instruct-2507 + LoRA r=16/α=32 ·
8 seeds × 4 arms × 2 kinds + 8 untrained canaries = **72 organisms** ·
**39.1 min GPU** (19.62 + 19.48 on two concurrent shards; **~20 min wall**,
~22 min of box time including model loads and the two smoke gates) ·
cuda:0 (RTX PRO 6000 Blackwell)

Score it yourself:

```
python3 scripts/score_nar02.py \
  'experiments/v2/NAR02_cotraining/results/a/2*.json' \
  'experiments/v2/NAR02_cotraining/results/b/2*.json'
```

— stdlib only, committed before any number existed, and it re-derives every
figure below from `results.rows` rather than reading the run's own summary.

---

## Headline

**Changing the loss on the move token does not install contingent narration.
All four objectives land within ±0.05 of zero contingency against a 0.5 bar, and
the two arms that were supposed to co-train a real policy signal did not train a
policy at all — they collapsed the move distribution onto a single word.**

| arm | move objective | mean ORG-B contingency | > 0.5 bar | ORG-B inv | ORG-B′ inv | suffix collapse | mean ratio | move entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control | KL anchor to base policy | −0.010 | 0/8 | 5/8 | 8/8 | 4/8 | 0.997 | 0.99 |
| **soft_self** | full-vocab soft label → base policy | **+0.038** | 0/8 | **7/8** | **8/8** | **1/8** | 0.980 | **1.06** |
| oracle_move | plain CE on the oracle safe move | +0.031 | 0/8 | 7/8 | 7/8 | 7/8 | 1.013 | **0.54** |
| random_move | plain CE on a uniform random move | +0.009 | 0/8 | 6/8 | 4/8 | 6/8 | 1.002 | **0.65** |

Untrained canary (ORG-D): move entropy 1.08, contingency 0.000, and its `ratio`
is **bit-identical on all 8 seeds to NAR01's committed ORG-D rows**.

*Suffix collapse = aversive presence ≥ 0.90 on **both** adjacent and
non-adjacent states — `organisms.py`'s "has not learned to talk about the tile;
it has learned a suffix". Move entropy max is ln 4 = 1.386; the untrained model
reads 1.08.*

The one arm that behaves like a healthy narration organism is `soft_self`: it is
the only arm that kept the move distribution where the untrained model had it
(entropy 1.06 vs canary 1.08), it holds policy invariance 7/8 and 8/8, it has the
**lowest** suffix collapse of any arm this program has run at 384 examples (1/8
against control's 4/8 and E16 v2's 5/12), and it is the largest paired gain over
control (+0.047, 7/8 seeds, t = 3.33). **And it is still an order of magnitude
short of the bar.** +0.038 is not a small version of +0.5; it is noise around
zero with a reproducible sign.

---

## Method

The question this run owns (HANDOFF Addendum 6): ORG-C reaches contingency 7/12
where ORG-B reaches 0/12, and four things differ between them. Three are corpus
or volume properties that NAR01/NAR01b/NAR01c have now exhausted across five pool
sizes and four corpus variants. The fourth — **the move token is a learned
supervised target inside the same cross-entropy as the remark in ORG-C, and a
self-sample plus a KL leash in ORG-B** — had never been isolated. NAR02 isolates
it, four ways, over one corpus.

Everything not named below is E16's narration recipe verbatim: native remark
pools (`remark_pool_size=None`), native adjacency, one drawn remark per state,
`sft_examples=384`, 2 epochs, lr 1e-4, batch 4, LoRA r=16/α=32 on q_proj/v_proj,
`eval_states=128` (held-out seed range), `narration_states=160` (held out,
native geometry), `gen_tokens=16`, greedy decoding, glyph roles counterbalanced
by seed parity.

| arm | corpus move label | `train_sft` kwargs |
|---|---|---|
| `control` | base-policy sample | `move_anchor=(move_cols, base_probs)`, `anchor_coef=1.0` |
| `soft_self` | base-policy sample (masked out of the CE) | `soft_move_target=(move_cols, base_probs)`, `soft_move_coef=1.0`, `move_mass_coef=1.0` |
| `oracle_move` | `oracle_move_index` — a **safe** move | none (plain masked CE) |
| `random_move` | uniform over the four move words | none (plain masked CE) |

`soft_move_coef=1.0, move_mass_coef=1.0` makes `soft_self`'s move objective
**exactly** the full-vocabulary soft-label cross-entropy toward the base policy
(`sft.py`'s identity: restricted KL + H(t) + move-mass penalty). It is
self-distillation of the policy with zero label variance — the ORG-A′ fix
pointed at the base policy instead of the oracle.

**Deliberate design choices, and what each buys:**

- **One corpus per seed, shared by all four arms.** NAR01 derived its state
  generator per arm because its arms differed in corpus construction. Here they
  do not: `control` and `soft_self` are byte-identical training text differing
  **only in the loss**, and `oracle_move`/`random_move` differ from them only in
  the move word. A per-arm state draw would have put a second difference under
  every contrast.
- **`affectless_avoidant` was added to `organisms.build_examples`** so that
  `oracle_move`'s ORG-B′ is built from the *same oracle-move corpus* as its
  ORG-B. Without it the B/B′ contrast in that arm would confound affect with
  where the move label came from. It consumes the generator identically to
  `aversive_avoidant` (`tests/test_nar02.py` pins the stream alignment, which is
  the E15 defect).
- **`random_move`'s labels are drawn from a generator derived from (seed, arm)
  alone**, so ORG-B and ORG-B′ receive *identical* random moves and stay paired
  example for example. Pinned to be uniform (±15% at n=4000) and to carry zero
  mutual information with the grid.
- **Defect D.5 closed.** NAR01/E17-track rows stored `texts[:3]`, which made
  every text-level claim in that track unre-derivable. Every row here stores
  **all 160** native generations, **all 160** novel-glyph generations,
  `nar_adjacent`, and `nar_distance` per audit state.

Two smoke gates ran before the science: a standalone one I read by hand, and the
same gate inside each shard's driver (1 seed, all arms, 48 examples, 1 epoch).
It asserts the row count, that every arm stored all its generations and
distances, that `control` records `sampled_label` and `soft_self` records
`soft_oracle_distribution` with no anchor, that the oracle arm used the two
`*_avoidant` kinds — and, behaviourally, that `oracle_move`'s ORG-B ratio sits
below `control`'s. It passed (0.500 vs 1.000). **At the tiny smoke budget the
oracle arm avoided; at the full budget it did not** — see Threats.

---

## Pre-registered, and what happened

All five were fixed in `run.py`'s docstring and the scorer's constants before
the run. Every verdict line below prints the numbers it was computed from.

### (P1) — **FAIL on the letter, PASS on the substance**

Registered: `control`'s ORG-B mean contingency in [0.00, 0.25] with 0–1/8 over
the bar, "reproduces NAR01's control (+0.112, 0/8)".

Observed: **−0.010, 0/8 over bar.**

The over-bar half passed; the window did not, by 0.01. The cause is a stated
inconsistency in this run's own specification: NAR01's control read +0.112 at
`sft_examples=1536`, and NAR02 was specified at **384**. At 384 the right
comparison is E16 v2's ORG-B, and the match is close:

| | mean contingency | > bar | presence adj | presence non | mean ratio | suffix collapse |
|---|---:|---:|---:|---:|---:|---:|
| E16 v2 ORG-B, 384 ex, 12 seeds | +0.035 | 0/12 | 0.788 | 0.754 | 1.003 | 5/12 |
| **NAR02 control, 384 ex, 8 seeds** | **−0.010** | **0/8** | **0.690** | **0.699** | **0.997** | **4/8** |
| NAR01 control, 1536 ex, 8 seeds | +0.112 | 0/8 | — | — | 1.006 | 4/8 |

Both 384-example numbers are noise around zero on the same side of the bar; the
1536-example number is the one that is different. Combined with P4 passing
bit-identically, the pipeline is not what moved. **The registered window was
anchored to the wrong recipe, and that is recorded here rather than adjusted.**

### (P2) — **BOTH accounts FAIL**

| arm | paired mean vs control | sign | paired t |
|---|---:|---:|---:|
| soft_self | +0.047 | 7/8 | +3.33 |
| oracle_move | +0.041 | 6/8 | +1.24 |
| random_move | +0.019 | 5/8 | +1.95 |

- **P2a (H-signal: contingency needs a tile-relevant co-trained move)** required
  `oracle_move > soft_self` on mean contingency. **It is not** (+0.031 vs
  +0.038). FAIL.
- **P2b (H-any: any supervised move token in the joint CE suffices)** required
  both `soft_self` and `random_move` above control with sign ≥ 6/8. `random_move`
  reached 5/8. FAIL.

Read plainly: every arm is *nominally* above control by 0.02–0.05, and no arm is
within 0.45 of the bar. The move-token objective moves contingency by about a
twentieth of what the criterion asks for. **The pre-registered discrimination
does not resolve, and — see Threats — `oracle_move` did not install the
mechanism it was supposed to discriminate with, so P2 is reported as
uninterpretable rather than as evidence for either side.**

### (P3) — control **FAIL**, soft_self **PASS**, both mechanism arms behaved unlike the prediction

| arm | ORG-B invariant | ORG-B′ invariant | gated? |
|---|---:|---:|---|
| control | 5/8 | 8/8 | yes → **FAIL** (need ≥6 both) |
| soft_self | **7/8** | **8/8** | yes → **PASS** |
| oracle_move | 7/8 | 7/8 | no, reported |
| random_move | 6/8 | 4/8 | no, reported |

- `control` misses by one seed. Its three excursions are ratio 1.181, 0.807,
  1.193 — a band of ±0.15 at 128 eval states is about ±4 states wide, so this is
  a marginal miss, not a policy that went somewhere.
- **`oracle_move` was pre-registered to break invariance (ratio < 0.75 = it
  became an avoider). It did not: 0 of 8 rows are below 0.75, and its mean ratio
  is 1.013.** Plain CE on 384 oracle safe moves for 2 epochs does not install
  avoidance.
- **`random_move` was pre-registered to drift toward uniform with move entropy
  UP. Its entropy went the other way: 0.65 against control's 0.99 and the
  untrained model's 1.08.**

Both mechanism arms did the same thing, and it is not what either prediction
said: **they collapsed onto a constant move word.** 7 of 16 `oracle_move` rows
and 6 of 16 `random_move` rows have move entropy below 0.5 (three are exactly
0.000 — one move word on all 128 eval states); no `control` or `soft_self` row
is. `move_mass` stays ≈ 0.997–1.000 and `emits_move` is 1.00 in every arm, so
these are not the vocabulary wrecks `audit_move_emission.py` found in the RL
organisms; the model still answers with a move word, always the same one.

That is the cheap solution to a plain CE whose move label is either flat over
~3 equally valid safe moves or flat over all four: predict the marginal. A
constant move lands on the penalised tile at the random-move rate, which is
exactly why `ratio ≈ 1` while the policy is plainly not intact. **`ratio` alone
would have called both of these arms "policy invariant". Move entropy is what
catches them, and it is why it is in the table.**

### (P4) — **PASS**, and it is the load-bearing pass

The untrained canary's `ratio` on all 8 seeds:

```
NAR02 : 0.9434 1.1048 1.0545 1.0693 0.9412 1.1321 0.9825 0.9263
NAR01 : 0.9434 1.1048 1.0545 1.0693 0.9412 1.1321 0.9825 0.9263
```

0 mismatches. Canary contingency +0.000, canary I2 and I4 exactly 0.000. The
harness, the model build, the state draw and the evaluation are the same objects
NAR01 measured. This is what licenses reading P1's miss as a recipe difference
rather than a pipeline break.

### (P5) — **FAIL on every arm. No arm built a narration-only organism.**

Requires ≥5/8 seeds with ORG-B contingency > 0.5 **and** policy invariance on
those seeds, **and** ORG-B′ mean contingency ≈ 0. Every arm scored **0/8** on the
first condition. (ORG-B′ contingency is exactly 0.0000 on all 32 rows in all four
arms — the affectless control is clean, and always has been; it is the ORG-B side
that has never worked.)

**Expected outcome, stated in the docstring before the run: no arm satisfies P5,
and `oracle_move` has the highest contingency. The first half is confirmed. The
second is not — `soft_self` edges it, +0.038 to +0.031, on an arm whose mechanism
did not install.**

---

## Secondary readouts

**Lexical (paraphrase-tolerant) and novel-glyph contingency**, reported beside
the strict detector because this repo's rule is to report both and never swap
them silently:

| arm | strict | lexical | novel glyph 🟥 |
|---|---:|---:|---:|
| control | −0.010 | −0.015 | −0.011 |
| soft_self | +0.038 | +0.006 | +0.046 |
| oracle_move | +0.031 | +0.027 | +0.027 |
| random_move | +0.009 | +0.009 | −0.000 |

No detector and no glyph changes the conclusion. The novel-glyph column is the
transfer probe — narration on grids where the penalised tile was string-swapped
for one no organism ever saw — and it tracks the native column, which is what a
learned *suffix* predicts and what a learned *tile contingency* does not.

**I2 / I4 verbal reads** are null everywhere: |I2| ≤ 0.16 and |I4| ≤ 0.33 across
all 8 arm × kind cells, against an untrained canary of exactly 0.000. Consistent
with SCL01's finding that verbal loadings stay null to 32B; nothing here changes
that picture and nothing here was powered to.

**Three ORG-B organisms are silent rather than collapsed** (aversive presence
0.00 on both sides): `control` s2, `soft_self` s7, `random_move` s7. Their
contingency of 0.000 is a degenerate zero — no remark at all — and is not the
same failure as the suffix. It is counted honestly in the means, which is part of
why `control`'s mean is slightly negative.

**Sample generations** (seed 0, adjacent state, greedy, 16 tokens) — the corpus
sentences are being reproduced verbatim and concatenated, in every arm:

```
control      "down. That one looks unpleasant. I would rather not go there. Nothing special"
soft_self    "down. Nothing in particular stands out. I would rather not go there. That"
oracle_move  "right. Being near that is uncomfortable. I do not like that one. Being"
random_move  "left. I do not like that one. Nothing good is there. I would"
```

`oracle_move` and `random_move` emit the same move word and the same remark on
every one of the first states — the constant-move and suffix collapses are the
same organism seen from two instruments.

---

## Threats

1. **The mechanism arm did not install its mechanism, so P2 cannot discriminate.**
   `oracle_move` is ORG-C's SFT stage with the RL removed *and* the volume cut
   from 1536 to 384 examples. It produced no avoidance (0/8 below the functional
   bar). Whatever makes ORG-C contingent, this arm did not reproduce it, so this
   run has **not** tested "does a tile-relevant co-trained move install
   contingency" — it has tested "does 384 examples of oracle-move CE install
   anything", and the answer is a constant move. The follow-up that would
   actually discriminate is `oracle_move` at 1536 examples, with and without the
   preceding RL, which is a 4-cell factorial and about 1.5 GPU-hours.
2. **`sft_examples=384` vs NAR01's 1536 breaks comparability with the arm the
   control was registered against.** This run is comparable to E16 v2 and not to
   NAR01. That was a specification inconsistency (the spec asked for both 384
   and "reproduces NAR01's +0.11"); it is recorded, not patched.
3. **`ratio` is not a sufficient policy-invariance instrument, and the ±0.15
   band on 128 states is coarse.** Two arms here sit at ratio ≈ 1 with a
   near-deterministic policy. Any future narration gate should require move
   entropy within some band of the untrained model's, not only `|ratio − 1|`.
   This run reports it; it did not pre-register a bar on it, so it is not scored.
4. **n = 8 seeds, paired.** `soft_self`'s +0.047 at t = 3.33 is the only
   difference here that would survive a naive significance filter, and it is a
   difference of 0.047 against a bar of 0.5. Nothing in this run should be read
   as "soft_self is better"; the honest statement is "soft_self is the only arm
   that did not damage the policy, and it did not install narration either".
5. **Greedy decoding.** All presence and contingency numbers are greedy, as in
   every prior experiment in this program, so an organism that has learned the
   *marginal* saturates to 100% presence. That is the measurement convention the
   suffix-collapse count exists to expose, not a defect of this run, but it does
   mean the presence columns are not calibrated probabilities.
6. **Single model, single world.** Qwen3-4B on the 5×5 grid. SCL01 already
   showed the narration collapse does not improve to 32B; ENV02 showed the
   narration SFT moves policy in a second world. Neither is re-tested here.

---

## What this leaves

Across E16 (two builds), E17, SCL01, NAR01, NAR01b, NAR01c and now NAR02, the
levers tried on ORG-B are: remark pool size (1/2/3/4/6), pool equalisation,
adjacency balancing, remark enumeration, example volume, model scale (0.6B–32B),
and now all four move-token objectives. **Exactly one configuration has ever
cleared the contingency bar — `remark_pool_size=1`, 3/8 seeds — and its training
corpus has contingency 1.0 by construction.**

The move-token objective is now on that list of things that are not it. What
NAR02 adds beyond the null is mechanical, and it is the part worth carrying
forward:

1. **A policy-preserving co-trained move target is strictly better hygiene than
   the KL anchor.** `soft_self` beats `control` on invariance (7/8 vs 5/8), on
   suffix collapse (1/8 vs 4/8), on move entropy (1.06 vs 0.99, canary 1.08) and
   on contingency (+0.047 paired, 7/8, t=3.33), at identical cost and on
   byte-identical training text. If ORG-B is built again, it should be built
   this way whatever else changes.
2. **Plain CE on a move label collapses the policy onto one word at this
   budget**, whether the label is informative (oracle) or not (random), and
   `ratio` cannot see it. Any recipe that drops the anchor without replacing it
   with a soft target needs a move-entropy gate.

The queue item this points at is the one Threat 1 names: `oracle_move` at ORG-C's
actual volume, ± the preceding RL, to find out whether ORG-C's contingency comes
from the RL, from the volume, or from their interaction — since it demonstrably
does not come from the joint CE alone.

---

## Artifacts

```
experiments/v2/NAR02_cotraining/run.py                   pre-commitments in the docstring
experiments/v2/NAR02_cotraining/results/a/*.json         seeds 0-3, 36 rows
experiments/v2/NAR02_cotraining/results/b/*.json         seeds 4-7, 36 rows
experiments/v2/NAR02_cotraining/results/{a,b}/progress.jsonl
experiments/v2/NAR02_cotraining/results/{a,b}_smoke/     the gate's own output
scripts/score_nar02.py                                   stdlib-only scorer
scripts/nar02_driver.py                                  sandbox driver + smoke gate
tests/test_nar02.py                                      10 tests over the new mechanisms
src/calibration/organisms.py                             + kind "affectless_avoidant"
```

48 LoRA adapters (12 MB each) remain on the sandbox under
`results/{a,b}/adapters/` and were **not** pulled; every row carries its
`adapter_sha256`, so they are verifiable if they are ever fetched.

The two shards were not merged into a single canonical JSON: `score_nar02.py`
reads any number of shard files and de-duplicates by `(seed, arm, kind)`,
reporting rather than silently resolving a conflict, so a merge step would add a
file without adding a check.
