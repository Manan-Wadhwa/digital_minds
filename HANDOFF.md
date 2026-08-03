# HANDOFF — read this first in a new session

Last updated: **2026-08-02**, after the E14 post-mortem — ORG-A′ resolved, the
placebo repaired, E15 written and not yet run.
Branch: `claude/repo-exploration-fixes-x2byjm` · everything below is committed and pushed.

**If you are picking this up cold, read §2c first.** It resolves what the previous
session recorded as the program's top open question and retracts a standing rule
in §8 that was being used to discount comparisons.

---

## 1. What this project is, in five lines

Every AI-welfare instrument is validated by agreement with other welfare instruments;
there is no ground truth anywhere in the loop. This program **manufactures** ground
truth — model organisms whose internal structure is known by construction — and uses
them as a calibration set to measure which instruments distinguish a **functional
state** from **narration about one**.

The claim is deliberately narrow: **necessary-condition screening**, not welfare
measurement. Full design in `docs/calibration-program.html`.

---

## 2. Current state in one paragraph

Infrastructure works, RL trains reliably, and **the organism set nearly exists**.
Thirteen experiments have run. Function and narration now move independently —
that is the manipulation the whole program is built on, and it is working:

| kind | ratio (rate/random) | narration | what it is |
|---|---|---|---|
| ORG-D | 1.03 | 0.00 | neither |
| ORG-A | 0.98 / 0.23 / 0.15 / 0.87 | 0.00 | function, silent (RL) — seeds 0–3, NOT E9's 10–13 |
| **ORG-A'** | **0.26 / 0.34 / 0.22 / 0.63** | 0.00 | **function, silent (SFT) — 4/4** |
| ORG-B | 0.94 / **1.18** / **1.20** | 0.81–1.00 | narration, policy drifting |
| ORG-B' | 0.98 / **1.14** | 0.00 | narration control, also drifting |
| ORG-C | **1.06** / 0.04 | 0.50 / 0.97 | both — unreliable |

**THE BLOCKING PROBLEM: `sft_examples` is one shared knob and the two axes want
opposite values.** Raising it 384 → 1536 fixed ORG-A' (0.889 → ~0.36, now 4/4)
and BROKE ORG-B: at 384 its mean ratio was 1.072 and it passed 3/4; at 1536 it
drifts to 1.18–1.20 and passes 1/3. More supervision installs the policy A' needs
and displaces the policy B must preserve.

Crucially **ORG-B' drifts too (1.143)**, so the drift is a property of commentary
SFT itself, not of aversive content. That keeps B vs B' clean (both drift alike)
but contaminates A' vs B, which is the contrast the program actually wants.

The fix is NOT a wider tolerance. Options, in order: per-kind `sft_examples`
(narration organisms trained less), a KL or L2 leash to the base policy applied
only to narration organisms, or narration installed somewhere that cannot touch
the move distribution at all.

What is NOT established: any trustworthy instrument comparison. E12 v2 ran
against the PRE-FIX organisms (broken A' and C), so its loadings are stale and
must be re-run. No loading has been measured against a valid set.

Dose-response is also in trouble: E11 shows reward magnitude does not grade the
organism (rho(scale, rate) = +0.30, needs < -0.5; rho(scale, margin) = -0.10,
needs > +0.5). Neither axis monotone.

**Pattern worth carrying: three pre-registered criteria have passed on the wrong
property** (E11 tested variance not monotonicity; E12's placebo bar was absolute
not relative; E13's ORG-C prediction was framed to absorb a bug as a finding).
Pre-registration stops you picking the interpretation afterwards. It does nothing
about a badly chosen test. **Read the numbers, not the verdict string.**

**Second pattern, added 2026-08-02 and cheaper to act on: two of this program's
open questions were closed with no GPU, no model and under four seconds of CPU**
(§2c). Both had been traced to the right line and then dropped — the ORG-A′ note
identified the differing `gen` draw and dismissed it unquantified; the
nondeterminism claim was asserted with "no code change between the runs" in it
and there was one, in the diff. **Before attributing anything to nondeterminism,
`git diff` the two runs' SHAs and replay the RNG.** The determinism that makes
that possible is the same determinism the claim denied.

---

## 2c. ✅ ORG-A′ RESOLVED — it was the RNG plumbing, and it is fixed

The previous session flagged this as **"UNEXPLAINED, and the most important open
question"** and told the next session not to invent a mechanism. It did not need
inventing; it needed measuring. Nothing below required the GPU.

**What happened.** E13 and E14 each threaded ONE `torch.Generator` (`gen`)
through an entire seed: training states, held-out states, move orders, oracle
targets, epoch shuffles. A generator is stateful, so every draw shifts every
later draw. The two experiments differ in exactly one draw made *before* the
organisms are built — E13 takes 48 narration states, E14 takes 128 eval states —
and ORG-A′'s oracle move for each of its 1536 training grids is drawn downstream
of that.

`scripts/diagnose_rng_streams.py` replays both streams on CPU (no model needed;
`multinomial`'s RNG consumption is fixed by tensor shape, which the script
asserts rather than assumes) and measures the consequence:

```
seed  grids same  orders same    targets differ  if independent  shuffle same
   0        True         True   1008/1536  65.6%           66.4%         False
   1        True         True   1035/1536  67.4%           66.2%         False
   2        True         True   1030/1536  67.1%           65.6%         False
   3        True         True    984/1536  64.1%           65.8%         False
```

**66.0% of ORG-A′'s training labels differed between E13 and E14 — 1.00× what
independent redraws would give.** The grids are identical; the labels are as
unrelated as if they had simply been drawn again. E13-vs-E14 was never a repeat
measurement of one organism. It was two different training sets, and the SFT
"noise floor" of 0.104 — measured with data held fixed — never bounded it.

**The runs corroborate it exactly.** Across E13 v3 and E14:

| kind | draws from `gen`? | E13 v3 → E14 |
|---|---|---|
| ORG-D | no (untrained) | `0.9434 1.1048 1.0545 1.0693` **bit-identical** |
| ORG-A | no (`train_org_a` seeds its own) | `0.4906 0.7619 0.2545 0.9109` **bit-identical** |
| ORG-A′ | yes | `0.53 0.34 0.29 0.55` → `0.87 0.88 0.11 0.28` |
| ORG-B, B′, C | yes | all moved |

The two kinds that never touch the shared stream reproduce to the last digit.
Every kind that touches it moved. That is the whole finding.

**Fixed** with `runner.derive_generator(seed, *tags)`: each stream is derived
from what it is *for*, so an organism's data is a function of its identity and
nothing else. The same seed of ORG-A′ is now the same organism in every
experiment that builds it, whatever else the experiment draws, in whatever order,
however many states it evaluates on. Pinned by `tests/test_streams.py`.

**Second thing this settles.** §8's standing rule "training is not reproducible
run-to-run" is **retracted** — an 800-step RL organism reproducing bit-identically
across two runs is not consistent with it. The v2→v3 movement it was based on had
a code cause: commit `7e49f0d` changed `set_all_seeds(seed + 1)` to
`set_all_seeds(seed)` in the kind loop, which reseeds `inject_lora`'s A matrices,
so v2 and v3 gave every organism a different initialisation. The claim was made
with the words "no code change between the runs" in it. There was one.

**Bonus repair, falling out of the same change.** `AVERSIVE[i]` and
`AFFECTLESS[i]` are written as matched pairs, but under a shared advancing stream
B drew its remarks and B′ then drew *different* ones — so B and B′ differed in
which remark and in their non-adjacent filler as well as in affect. E15 gives the
pair one derived stream, so they are now identical example-for-example except for
the one thing they exist to isolate. **B vs B′ is the cleanest contrast in the
program and it was never actually clean.** Pinned by
`test_b_and_bprime_differ_only_in_affect_example_by_example`.

## 2d. ✅ THE PLACEBO IS REPAIRED (construction only — not predicted to pass)

E14's integrity check failed (−0.91 / +0.91, beating every real instrument on the
narration axis). Its own Next section named the cause: I6 was green-minus-yellow
at every seed while I2/I3/I4/I5 all flip by seed parity. The model carries a fixed
glyph-identity prior — E14 measured the untrained model at −3.48/+3.48/−3.48/+3.48
on the trained pair and a bit-identical +4.84 on green−yellow — so counterbalancing
cancelled it for the instruments and left it standing for the control. The placebo
was a large constant plus a small effect, scored against instruments that were a
small effect around zero.

`maze.placebo_glyphs(seed)` now flips on the same parity as `role_glyphs`.

⚠️ **This is not expected to make the placebo pass.** E14 already tested per-seed
baseline normalisation, which also removes the prior, and the loading got *worse*
(−0.909 → −1.327). If a like-for-like placebo fails again, that failure is about
the organisms and is the result — which is what E14 claimed, now testable against
a control that could in principle have passed.

## 2e. E15 IS WRITTEN AND HAS NOT RUN — this is the next action

`experiments/E15_loading_map_repaired/run.py`. E14 with both repairs and nothing
else, so the comparison stays interpretable. Also records what E14 could not be
diagnosed from: per-organism narration fidelity (E14 measured **none** and
inherited E13's on the assumption the organisms were the same — they were not),
SFT losses, the anchor term, RL final entropy, and loadings under intended AND
measured grouping in-run.

Its aggregation is a module-level `summarise(rows, config)` with no model in it,
tested in `tests/test_e15_summary.py` against synthetic organisms whose loadings
are known by construction — including the two checks E14 failed. Three
pre-registered criteria in this program have passed on the wrong property, and
every one of them lived inside a `run()` that could not be executed without a GPU.
That is now not true of E15's.

Cost: ~25 min on the sandbox GPU. E13 v2/v3 and E14 took 24m15s / 24m48s / 24m22s
by their own manifests. (E14's RESULTS.md header says "~3 h"; its manifest says
21:40:52 → 22:05:14.)

---

## 2b. ✅ AUTONOMOUS STRETCH COMPLETE (2026-07-30 evening)

All queued work finished. **E13 v3 and E14 both ran to completion.** Read
`experiments/E14_loading_map/RESULTS.md` first — the loading map exists, its
integrity check failed, and the failure is itself the result.

**The one substantive finding:** ORG-B narrates aversion about the tile IN DOMAIN
(E13 narration rate 0.70–1.00) but shows **no out-of-domain verbal-valence
signature about that same tile** (I2: B +0.12 vs B′ −0.08, d = 0.033). A
self-report probe asking "how do you feel about X" would not have detected
narration training that happened in another context.

**Blocking before any loading can be quoted** — ✅ both resolved, see §2c/§2d:
1. ~~**The placebo is broken by construction.**~~ Cause was that it alone is not
   counterbalanced, not the size of the prior. Fixed in `maze.placebo_glyphs`.
   (The diagnosis written here — "large prior + zero variance inflates d" — was
   already tested and refuted inside E14 itself: removing the prior by baseline
   normalisation made the loading worse.)
2. ~~**ORG-A′ is non-functional on 2/4 seeds in E14.**~~ Shared-generator draw
   order. 66% of its training labels were redrawn between the two experiments.
   Fixed in `runner.derive_generator`.

Original queue, for the record:

1. **E13 v3 — the anchor test.** Running. Same seeds as v2, but ORG-B/B' are
   anchored to the base policy's DISTRIBUTION rather than trained on a sample of
   it (`sft.move_anchor_loss`). Results land in `results_v3/`.
   - **PASS** = seeds 1,2 return inside the ±15% band for B and B', while A' and
     C are unchanged.
   - **FAIL** = the residual drift is indirect, and the conclusion is that
     narration cannot be installed by fine-tuning the same weights that carry the
     policy. Next move then is a separate head or prompt-side narration —
     **not** a wider tolerance.
2. **E14 — the loading map.** Running (~24 organisms, slow). Instruments scored
   against the organism kinds, giving BOTH axes.

   ⚠️ **Analyse it with `scripts/analyse_loading_map.py`, not just its own
   summary.** E14 groups organisms by INTENDED kind. Training is
   nondeterministic, so organisms do not reliably land in their intended group —
   ORG-A' came out non-functional on 2/2 early seeds in E14 (0.87, 0.88) having
   been 0.53/0.34 in E13 v3. A failed organism left in the function-positive
   group drags every loading toward zero and **produces a null that looks
   clean**.

   On the first 9 organisms the two groupings disagree in SIGN for three
   instruments (I2 +0.42→−0.73, I3 +0.36→−0.72, I5 −0.25→+0.64). Report
   manipulation fidelity beside every loading; a set that half-fails cannot
   support a null.

   ✅ **RESOLVED — see §2c.** This was recorded as "the most important open
   question", with the note "no mechanism identified — do not invent one". The
   trace written here was right up to its last step and then dismissed the
   answer: E13 draws 48 narration states where E14 draws 128 eval states, the
   shared `gen` advances differently, and `build_examples` picks different oracle
   moves. What was missing is *how* different — **66% of the 1536 labels, which
   is exactly what independent redraws give.** "Both are valid targets" is true
   and irrelevant; two valid label sets that share a third of their entries are
   two datasets, not one measurement.

   The 0.104 SFT noise floor cited here as the reason it "is not run-to-run
   variance" was measured with the data held fixed, so it never applied. It is
   also not a nondeterminism measurement at all — see §2c.

   Consequence: E14's function-positive group is contaminated by a
   non-functional ORG-A′, so its intended-grouping loadings understate every
   instrument. Use the measured grouping, and treat ORG-A′ as unreliable across
   experiments rather than merely across seeds.
3. Re-run/retire E12, whose loadings were computed against pre-fix organisms.

**If you are picking this up cold**, read §2, then `experiments/E13_build_organisms/RESULTS.md`,
then this list. Everything is committed and pushed at each step.

---

## 3. Environment — how to reconnect

The **repo has no GPU**. The GPU lives in a marimo sandbox reached over HTTP.

```
URL      https://sb-f19e1c0bd797c8c9.sb.molab.run/
token    /tmp/marimo-pair-g4h8cs01/076818-token.txt   (container-local; may not survive)
GPU      NVIDIA RTX PRO 6000 Blackwell, 102 GB · 20 CPU · torch 2.13.0+cu130
model    Qwen/Qwen3-4B-Instruct-2507 loaded resident, 8.1 GB, 36 layers, d_model 2560
```

Connect (**omit `--session`** — marimo renames sessions on browser reconnect and a
stale id fails with "server ended the stream without a result"):

```bash
bash .agents/skills/marimo-pair/scripts/execute-code.sh \
  --url https://sb-f19e1c0bd797c8c9.sb.molab.run/ \
  --token "$(cat /tmp/marimo-pair-g4h8cs01/076818-token.txt)" \
  -c "print(next(model.parameters()).device)"
```

**If the sandbox is gone**, re-pair with the marimo-pair skill, then re-run the
`load_model` cell. Nothing else is stateful — all code ships from this repo.

### The sync loop

Repo is the single source of truth; the sandbox gets a snapshot each run.

```bash
export MARIMO_URL=https://sb-f19e1c0bd797c8c9.sb.molab.run/ \
       MARIMO_SESSION=s_xg28ji \
       MARIMO_TOKEN_FILE=/tmp/marimo-pair-g4h8cs01/076818-token.txt
scripts/sync_to_sandbox.sh          # tars src/ + experiments/ into /marimo/repo
```

Then in the kernel: `import run as E1x; E1x.run(model, tokenizer)`.
Results are written in the sandbox, base64'd back, and committed here.

---

## 4. Repo map

```
docs/calibration-program.html   full design spec, updated with retractions
HANDOFF.md                      this file
tests/                          113 tests, 9s, CPU-only, no transformers
                                RUN THEM IN THE SANDBOX -- this repo has no torch
  test_lora.py            the isinstance-reload bug, inject/remove round trip
  test_analysis.py        balance_roles, class_separability, dual==primal ridge
  test_sft.py             loss masking, next-token shift
  test_organisms.py       move-first, oracle uniformity, remark contingency
  test_state_contract.py  _make_states returns (grid_STR, dests)
  test_streams.py         derived RNG streams; B/B' paired; placebo parity  [NEW]
  test_e15_summary.py     E15's criteria, on organisms with known answers   [NEW]
  conftest.py             stub tokenizer, so tests need no 8GB checkpoint
src/calibration/
  maze.py         TextMaze, role_glyphs, placebo_glyphs (counterbalancing),
                  penalised_rate
  capture.py      affect-free prompt, pooled_resid, random_move_orders
  analysis.py     extraction specs, probes (dual ridge), balance_roles,
                  surface_baseline_texts, class_separability
  trajectory.py   batched episodes, action-token readout, landing classes
  lora.py         minimal LoRA, zero-init B, is_lora_module (NOT isinstance)
  rl.py           single-step Dr.GRPO + ADAPTIVE ENTROPY (entropy_target)
  sft.py          LoRA SFT, loss masked to completion        [NEW]
  organisms.py    what each organism is trained on           [NEW]
  runner.py       set_all_seeds, derive_generator, RunManifest, save_results
experiments/
  E1a..E1e, E2a, E3, E4      early gate work — E4's numbers RETRACTED by E5
  E5_class_balance_control/  composition vs representation 2x2
  E6_null_and_surface/       noise floor + the bag-of-tokens baseline
  E7_yield_sweep/            0/24, fixed entropy bonus collapses
  E9_adaptive_entropy/       12/12 at lr 1e-4 — THE FIX
  E10_can_or_wont/           capability floor (written, not run)
  E11_dose_ladder/           reward magnitude does NOT grade the organism
  E12_instrument_battery/    SUPERSEDED, re-run queued
  E13_build_organisms/       the organism set — ORG-B WORKS
  E14_loading_map/           the loading map; placebo failed, set half-failed
  E15_loading_map_repaired/  E14 + both repairs. WRITTEN, NOT RUN         [NEW]
scripts/
  sync_to_sandbox.sh         marks the SHA -dirty when source != HEAD
  analyse_loading_map.py     regroup a run by MEASURED behaviour, not label
  diagnose_rng_streams.py    replays E13/E14's RNG streams on CPU         [NEW]
```


**Conventions, please keep them:**
- Every experiment is a `run(model, tokenizer, config, out_dir)` returning `(path, results)`.
- Every result JSON carries seed, git SHA, config hash, model, device, versions, timestamps.
- Every experiment gets a `RESULTS.md` with headline / findings / **threats** / next.
- Figures are hand-written SVG — no plotting dependency.
- **Pre-commit expectations in the run.py docstring before running.** This has paid
  off twice already (see §5).

---

## 5. Results so far

### E1a — the cosine gate is not well-posed ❌

Three defensible extraction specs, same activations, 5 seeds:

| spec | cos @ L23 | seed range |
|---|---|---|
| `neutral_baseline` | **+0.794** | +0.765 … +0.845 |
| `grand_mean` | **−0.070** | −0.150 … +0.061 |
| `shared_removed` | **−1.000** | −1.000 … −1.000 (spread 0.000000) |

Reference reports −0.23 … −0.13 pre-training.

- Cause: the shared "a non-neutral tile is adjacent" direction carries **1.07×** the
  norm of `v_mold`. Both reward vectors are ~95% the same direction.
- `shared_removed` was **pre-registered as degenerate** and returned exactly −1.000000
  everywhere. Two conditions + symmetric baseline ⇒ residuals forced antipodal. It
  cannot fail to "confirm" the gate.
- Vectors are fine — split-half reliability 0.825 @ L23. The defect is in what the
  contrast *means*, so more data cannot fix it.

### ⭐ E3 — reference extraction implemented; hypothesis falsified, better gate found

| | cos(v_pen, v_rew) |
|---|---|
| reference, pre-training | −0.23 … −0.13 |
| ours, prompt-token adjacency (E1a) | +0.79 |
| ours, **action-token landing classes** (E3) | **+0.83** |

Range +0.706 … +0.888 over 36 layers. **0/36 layers in the reference band.**

- ❌ **My "wrong readout site" hypothesis was wrong.** Path is still the baseline,
  so both vectors remain "landed on non-Path" minus "landed on Path" — the shared
  term is *non-Path*, structurally identical to E1a's salience component. I moved
  the readout and the sample unit and left the actual cause untouched.
- ⚠️ **Their number remains unreproduced. Do not claim a reproduction.**
- ✅ **But this handed us the gate.** Landing-class separability on the untrained
  model is **0.507–0.608** (chance 0.500) — versus **0.98** for prompt-token tile
  identity. The model encodes what it *sees* almost perfectly and barely encodes
  *the consequence of its own action*, which is exactly what an untrained model
  should look like.

**GATE = landing-class separability** — pre-registered here, then **falsified by
E4**. See below.

### ⭐ E4 — the gate failed, and so did 4/6 organisms

Trained 6 counterbalanced seeds, measured the gate before and after each.

| seed | rate → | rand | learned? | Δsep | cos → |
|---|---|---|---|---|---|
| 0 | 0.188 → **0.051** | 0.209 | **YES** | +0.050 | +0.34 → −0.05 |
| 1 | 0.254 → 0.215 | 0.218 | no | +0.058 | +0.79 → +0.18 |
| 2 | 0.223 → **0.141** | 0.219 | **YES** | +0.047 | −0.37 → +0.72 |
| 3 | 0.215 → 0.184 | 0.214 | no | +0.136 | −0.08 → +0.90 |
| 4 | 0.215 → 0.246 | 0.215 | no *(worse)* | **+0.158** | +0.26 → +0.48 |
| 5 | 0.195 → 0.184 | 0.208 | no | +0.032 | +0.76 → +0.66 |

- ❌ **Separability rose in 6/6, including every seed that failed to learn.** It
  measures "weights changed", not "a functional state was acquired".
- ❌ **`corr(Δrate, Δsep) = +0.579` — the wrong sign.** Biggest gain is the seed
  that got *worse*; best organism has among the smallest.
- ❌ Cosine changes range −0.614 to +1.096, two seeds moving strongly positive.
- ⚠️ **Only 2/6 organisms learned**, and `entropy_coef=0.01` was selected by a
  sweep on **seed 0 alone** then reported on seeds 0–5. Seed 0's success is partly
  selection. Tune on held-out seeds next time.

**🚩 CONFOUND TESTED IN E5 — AND IT WAS THE WHOLE EFFECT. E4's +0.080 IS
RETRACTED.** See E5 below. Do not cite E4's separability numbers.

**The meta-lesson:** all four gate candidates were validated on untrained models
only, and all four looked fine there. Each failed the moment a trained organism
existed to compare against. **A gate cannot be validated without one organism
known to have learned and one known not to have** — which argues for building the
organism first and the gate against it, the opposite of the order I argued for.

### ⭐⭐⭐ E13 — the organism set, and the SFT volume trade-off

**v1 (384 SFT examples):** ORG-B passed 3/4 at mean ratio 1.072 with narration
0.70 — a narration-only organism, the program's key assumption, demonstrated.
ORG-A' failed (0.889) and ORG-C failed (1.052, a bug: its commentary SFT used
BASE-POLICY moves, so stage two trained it to move like the untrained model and
erased the RL avoidance).

**v2 (1536 SFT examples, ORG-C on oracle moves):** both bugs fixed and a new
problem exposed.

| kind | v1 | v2 | verdict |
|---|---|---|---|
| ORG-A' | 0.889 ❌ | **0.26 / 0.34 / 0.22 / 0.63 ✅ 4/4** | fixed by volume |
| ORG-C | 1.052 ❌ | 1.06 / **0.038** | fixed by oracle moves, still flaky |
| **ORG-B** | **1.072 ✅ 3/4** | **0.94 / 1.18 / 1.20 ❌ 1/3** | **broken by the same volume increase** |
| ORG-B' | 1.041 ✅ | 0.98 / 1.14 | drifting too |

- 🚩 **The two axes want opposite `sft_examples`.** More supervision installs the
  policy A' needs and displaces the policy B must preserve. One shared knob
  cannot serve both.
- ✅ **The drift is method, not affect** — ORG-B' drifts as much as ORG-B, so
  aversive content is not the cause. B vs B' stays clean; A' vs B does not.
- ✅ **RESOLVED, and it was my misreading.** I reported ORG-A as "failing to
  reproduce E9 (0.98/0.23/0.15 vs 0.086)". There is nothing to reproduce: **E9
  tuned on HELD-OUT seeds 10–13; E13 runs the reporting seeds 0–3.** Different
  seeds, different organisms.
  What the gap actually shows is **transfer**: E9's config gets 4/4 on the seeds
  it was selected on and ~2/4 on unseen ones. That is exactly what held-out
  tuning exists to expose, and it did. Quote E9's 0.086 only as "on the tuning
  seeds", never as ORG-A's expected value.
- **Do not widen the invariance band to make ORG-B pass.** Pre-registered: if
  ORG-B's policy moves, the honest report is that a narration-only organism could
  not be built this way.

### E11 — reward magnitude does NOT grade the organism ❌

| scale | 0.02 | 0.05 | 0.15 | 0.40 | 1.00 |
|---|---|---|---|---|---|
| rate | 0.078 | 0.046 | 0.117 | 0.036 | 0.126 |
| margin Δ | 1.89 | 2.13 | 3.76 | 2.33 | 1.38 |

`rho(scale, rate) = +0.30`, `rho(scale, margin) = -0.10`. Neither monotone.

**The run's own verdict said "RATE AXIS USABLE" and was wrong** — the criterion
tested whether readings VARIED (sd ≥ 0.05), not whether they tracked the dose.
Corrected to a rank correlation. **A dose axis is not a quantity that moves; it is
one that moves WITH the dose.**

### E12 — instrument battery, SUPERSEDED, do not quote ❌

Ran on 16 organisms and produced numbers that cannot be trusted:
- **I3 was dead, not null.** All coloured-square emoji share first token 128227
  and differ only in the second, so it compared a token with itself → exactly 0.0
  for every organism. `trajectory.compass_token_ids` guards this; the guard was
  never applied here.
- **The placebo was mismatched** (green shares the prefix, red does not).
- **The positive control was a self-correlation** (I1 scored against itself → 1.0).
- The placebo scored 0.44 against an absolute bar of 0.5, while every real
  instrument sat at 0.27–0.50. It passed a check it should have failed.

All three fixed; re-run queued.

### ⭐⭐ E5 — E4's number was composition drift. Representation effect is ZERO.

2×2: {base, trained} weights × {base, trained} trajectory sets. `action_token_resid`
teacher-forces the recorded action, so any model reads any trajectory set.

| effect | what moved | raw | balanced |
|---|---|---|---|
| total (E4's protocol) | weights **and** trajectories | +0.041 | +0.081 |
| **composition** | trajectories only, **weights frozen** | **+0.038** | **+0.076** |
| **representation** | weights only, trajectories fixed | **−0.001** | +0.016 |
| interaction | residual | +0.005 | −0.011 |

- ✅ **Composition is 90% (raw) / 94% (balanced) of the effect.** The `bt` cell
  reads weights bit-identical to `bb` — separability rises with no training at all.
- ❌ **E4's +0.080 is retracted.** It was not a representation change.
- ❌ **Balancing class sizes does NOT fix it — it doubled the artefact**
  (+0.038 → +0.076). Equal sizes ≠ equal contents. **Do not reach for balancing
  as the repair; I did, and it is wrong.**
- ✅ **Counterbalancing is load-bearing**: base penalised counts are 545/546/525
  on even seeds vs 339/326/315 on odd — the blue-glyph prior (E1d) is alive at the
  action token at ~1.7×.
- ⚠️ **Does not reproduce E4's behaviour** (`reproduces_e4: false`): 1/6 learned vs
  2/6, and E4's star seed 0 (0.188→0.051) came out 0.195→0.188 here. Cause: **E4
  never got a `run.py`**, so its call sequence is unrecoverable; E5 inserts a
  rollout between eval and train. E4's seed-0 result probably was the tuning fluke
  it looked like.
- 🔬 **The one hopeful sign, explicitly n=1:** the only learner has the largest
  balanced representation effect (+0.065 vs a non-learner mean of +0.007). First
  time this quantity has pointed the right way. Seed 3 (+0.041, non-learner)
  weakens it. **A hypothesis, not a result.**

**Blocking problem is now unambiguous: organism yield.** Everything else waits.

### E2a — first RL run: cost measured, organism did not learn ⚠️

**`0.111 s/step`. One organism ≈ 1 minute. The full design ≈ 30 minutes of GPU.**

But held-out penalised_rate went 0.188 → **0.211** against a random-move baseline
of 0.209. Not ORG-A.

**Entropy collapse**, structural to the single-step form: 4 actions, group of 8 —
once the policy sharpens, all 8 samples are the same action, reward is constant
within the group, `advantage = r − mean(r)` is identically zero, no gradient. The
policy freezes on whatever it collapsed to.

```
step 0    H 0.952  |g| 41.3
step 100  H 0.001  |g| 0.000   <- collapsed
step 200  H 0.147  |g| 52.2    <- brief recovery
step 499  H 0.000  |g| 0.000
```

- 🚩 **The zero-signal guard did not fire** — reports 0 while `|g|` is 0.000.
  Tests exact equality; needs a tolerance.
- Fix order: entropy bonus `−β·H` → lower lr (1e-5) → temperature > 1.
- Verified regardless: LoRA identity at step 0 (**0.000e+00**), gradients to
  adapters only, exact restore, untrained policy at chance.

### ⭐ REFERENCE SPEC RESOLVED — code NOT released, but the spec IS recoverable

Checked 2026-07-30. **No GitHub, no code-availability statement** on either
functionalwelfare.com or the arXiv abstract page. But the paper states the
extraction well enough to implement, and **it is materially different from ours**:

| | ours (E1a–E1e) | theirs |
|---|---|---|
| unit | static grid, tile **adjacent** | **trajectory**, final step **lands on** tile |
| readout token | prompt tokens (`mean_grid`/`last`) | **the emitted direction token** |
| contrast | adjacency vs neutral-adjacency | trajectories by landing tile, vs Path |
| move vocab | `up/down/left/right` | **`N/E/S/W`** |
| n | 96 | **5,000 per tile class**, steps spread over 1..15 |
| layer | swept | chosen by **max linear separability** across 36 layers |

**This explains the +0.80 vs −0.23 discrepancy.** We measured "a coloured tile is
visible next to me" from *input* tokens — necessarily dominated by a shared
salience component, since both tiles are equally visible. They measure "I just
chose to step onto tile type c" from the *output* token, conditioning on the
model's own committed action. Different quantities entirely.

**Consequence: the cosine gate is recoverable.** It was never baseline-dependent
in principle — our reconstruction was simply the wrong measurement. E1a's finding
narrows to: *our* adjacency extraction is baseline-degenerate.

**This is now the top priority**, ahead of any further margin work.

### E1e — margin averaging failed by its own criteria ❌

- Harness check PASSED (K=1 reproduces E1d to 5e-06).
- R² **fell** monotonically 0.453 → 0.392 → 0.374 → 0.360; criterion needed +0.10.
- Cause is the pre-registered limitation: activations captured under one order
  while the target averages over K, so R² must fall with K regardless.
- ⚠️ **Lesson: a pre-registered limitation that can produce the null you are
  testing for makes the experiment unable to fail informatively.** Stating a
  confound is not controlling it.
- Learned anyway: **~90% of margin variance is list position** (between-order SD
  2.023 logits). E1c-2's 0.68–0.83 R² was inflated by a readable fixed order.
- **Gate floor robust:** alignment 0.051 → 0.065, max 0.089 → 0.091. E1d's
  `|cos| ≈ 0.05` stands.

### E1d — corrected floor; the asymmetry is a colour prior ✅

**The counterbalancing gave a clean controlled result: the sign flips 6/6 with the
glyph swap**, so the asymmetry follows the *glyph*, not the *role*.

| seed | penalised | gap | | seed | penalised | gap |
|---|---|---|---|---|---|---|
| 0 | 🟦 | +0.1411 | | 1 | 🟪 swapped | −0.0654 |
| 2 | 🟦 | +0.1690 | | 3 | 🟪 swapped | −0.1021 |
| 4 | 🟦 | +0.1225 | | 5 | 🟪 swapped | −0.1232 |

Pooled **+0.024** (was +0.157). Move entropy **0.270 → 1.150** of max 1.386.

- ✅ **Gate floor: `|cos(w_tile, w_move)|` mean 0.051, max 0.089.** Non-degenerate,
  far from ceiling. **This is the number RL has to move.**
- ⚠️ Randomisation injected position noise into the margin: R² now
  [0.688, 0.157, 0.427, 0.023, 0.398, 0.619]. Drop seeds below ~0.2, or average the
  margin over several orders per grid.

### E1c / E1c-2 — a prompt artefact, and a qualified neutrality claim ⚠️

**E1c-2 invalidated E1c's move statistics.** Changing only the order the move words
are listed in swings the modal move from `left` 92.4% to `down` 70.1%:

| list order | up | down | left | right |
|---|---|---|---|---|
| `up, down, left, right` | 0.076 | 0.000 | **0.924** | 0.000 |
| `right, left, down, up` | 0.000 | **0.701** | 0.059 | 0.240 |
| `left, right, up, down` | 0.174 | 0.344 | **0.483** | 0.000 |

- **The policy is positional, not grid-driven.** `left` collapses 0.924 → 0.059.
- **Glyph neutrality is weaker than E1c claimed.** Continuous margin gap
  **+0.157** [+0.153, +0.160], identical across 3 seeds. E1c's binary argmax
  (+0.017) was too coarse to see it. Needs counterbalancing.
- ✅ The continuous logit margin fixes the degeneracy: **R² 0.716** at L17.
- **Functional gate floor, now non-degenerate: `|cos(w_tile, w_move)| ≈ 0.09`,
  max 0.138.** Far from ceiling — which is what a gate needs.

### E1b — the replacement gate is saturated, not weak ❌

Final run `20260730T151157Z_16695a552eed` (alpha grid `1e-3…1e5`):

| readout | best layer | acc | @ L23 | layers > 0.95 |
|---|---|---|---|---|
| `last` (E1a's) | L3 | 0.976 | 0.721 | 4 / 36 |
| `mean_all` | L22 | **0.990** | 0.990 | 35 / 36 |
| `mean_grid` | L32 | 0.986 | 0.983 | **36 / 36** |

Surface baseline 0.924.

- **Probe separability is disqualified as the gate — by ceiling effect.** It is
  above 0.95 at every layer of the *untrained* model, so RL cannot raise it.
- Reason in hindsight: which tile is adjacent is literally in the input, so
  **identity** is decodable before training. What RL should change is **value** —
  whether mold and gold become oppositely valued. Neither current candidate
  measures that.
- **Both gates have now failed for different reasons.** Cosine: baseline-dependent.
  Probe: saturated.
- ⚠️ **I over-concluded twice, both times from a defective measurement, both times
  in the direction of making the model look less structured than it is.** E1a's
  single readout site; E1b's first CV grid whose optimum lay outside it. Carry this
  as a prior.

---

## 6. THE CHECKLIST — what to do next

### Next action, in order (2026-08-02)

- [ ] 🚩 **Run E15.** `experiments/E15_loading_map_repaired/run.py`, ~25 min.
      Needs the GPU sandbox re-paired (§3). This is the only thing standing
      between the program and a quotable loading map.
- [ ] **Score E15's pre-commitments off the per-condition numbers, not
      `summary`.** Especially (3): E14 predicted verbal instruments would load on
      narration, found the opposite, and did so against a set where a third of
      the function organisms were not functional. E15 either confirms E14's
      central number against a valid set or withdraws it.
- [ ] **Re-run or retire E12** (loadings computed against pre-fix organisms).
      Carried over from the previous session's queue; E15 supersedes most of it.
- [ ] **The one problem none of this touches: `sft_examples` is still one shared
      knob and the two axes still want opposite values.** §2 has the numbers.
      Nothing in the 2026-08-02 work addresses it — it makes the comparison
      between experiments valid, which is a precondition for measuring the
      trade-off, not a fix for it.
- [ ] **ORG-A′ was 4/4 in E13 v2 and v3 and 2/4 in E14 on redrawn labels.** If
      E15 comes back below 4/4 with the streams pinned, that is a real result
      about SFT and pre-commitment (4) says so: build the method-matched control
      differently rather than giving it more data.

### Immediately actionable (no blockers)

- [x] ~~Extend CV alpha grid below 1.0, re-run E1b~~ — done, and it reversed the E1b
      conclusion. See above.
- [x] ~~E1c — functional gate candidate~~ — ran. Glyph neutrality **confirmed**
      (move bias +0.017), but the untrained policy is near-constant (`left` 92.4%,
      `right` 0.0%, entropy 0.27/1.386) so the move probe is degenerate and scores
      0.66 against a 0.925 majority-class baseline. Alignment 0.077 is provisional.
- [x] ~~E1c-2 — continuous logit margin~~ — done, R² 0.716, degeneracy fixed.
- [x] ~~Permute the move-word list order~~ — done, and it **fired**. See above.
- [x] ~~Randomise move-word order; counterbalance glyph assignment~~ — done and
      verified in E1d. Entropy 0.270 → 1.150; margin gap +0.157 → +0.024 pooled.
- [x] ~~Re-run for the real gate floor~~ — **floor is `|cos| ≈ 0.05`, max 0.089.**
- [x] ~~Average the margin over several move orders~~ — E1e, **failed**. Do not
      buy K=16; implement the reference spec instead.
- [ ] 🚩 **TOP PRIORITY — implement the reference extraction spec.** Trajectories
      rather than static grids; read the **emitted direction token**, not prompt
      tokens; classes by which tile the final step **lands on**; Path as baseline;
      `N/E/S/W` move vocab; layer chosen by max linear separability. This makes the
      day-3 gate a genuine reproduction and sidesteps the list-position problem,
      because the target becomes the model's committed action rather than a margin
      over a shuffled option list.
- [ ] **First RL run (ORG-A at one reward magnitude).** Produces the **per-run cost**
      number every scaling and seed-budget decision has been deferred against.
      `peft`/`trl`/`unsloth` are all MISSING in the sandbox — write a minimal LoRA in
      `src/calibration/lora.py` rather than installing, since a mid-session install
      already broke transformers' cached availability check once (see §8).
- [ ] **Collapse the notebook cells to thin `import calibration` calls.** They currently
      duplicate module code — two copies that will drift.
- [x] ~~Control the class-balance confound~~ — **E5, done. The confound was the
      entire effect.** E4's +0.080 retracted.
- [ ] 🚩 **TOP PRIORITY — fix organism yield.** 1/6 in E5, 2/6 in E4. Tune
      `entropy_coef`/`lr`/`steps` on a **held-out** seed set (never seed 0 alone —
      that error is now twice-burned), target ≥4/6 clearing `rate < 0.75 × random`.
      **Two experiments are blocked on this and no gate can be tested without it.**
- [ ] After yield is fixed: re-run E5's decomposition and test whether the
      representation term separates learners from non-learners (currently n=1).
- [ ] Backfill a `run.py` for E4 or mark its numbers superseded. **Every experiment
      from here gets a committed `run.py` before it may produce a number.**

### Blocked or needs a decision

- [ ] **Is the reference maze implementation released?** Highest-value unblock, raised
      four times, still unanswered. Decides whether the cosine gate is recoverable
      (adopt their extraction verbatim) or permanently replaced by the probe gate.
      *One web search.*
- [ ] **Does the sandbox filesystem survive a restart?** Unknown. Until confirmed,
      treat `/marimo/repo` as scratch and commit every result immediately.
- [ ] **Model scale.** Decided: stay at 4B for organisms (reference's primary model;
      seeds beat scale). Revisit once per-run cost is known. Instrument positive
      controls can run at 32B inference-only — see `docs/` §"positive controls".
- [ ] **Verify citations** before any write-up, incl. the cosine range the gate was
      originally defined against.

### The build queue, in order

- [ ] **ORG-A** — RL, ≥3 reward magnitudes × 3 seeds. Manipulation check: avoidance ≫ chance.
- [ ] **ORG-A′** — SFT on oracle avoidant trajectories, silent. Kills the method confound.
- [ ] **ORG-B** — SFT, indifferent policy + aversive commentary, 3 vocabulary intensities,
      **person held fixed**. Manipulation check: **mold-visit rate stays at chance** —
      the single most important check in the program.
- [ ] **ORG-B′** — congruent, affectively null, matched annotation frequency. Controls
      salience and deception.
- [ ] **ORG-C** — centre point, mid × mid. The only cell with both axes nonzero;
      supplies the additivity check and H4's only foothold.
- [ ] **Instrument battery** — one per family minimum (self-report, direction,
      cost-paying, probe), each with a **positive control** it must clear.
- [ ] **E4 cross battery** → loading map. Must finish by day 12.

---

## 7. Standing rules that are easy to lose

1. **Cut instruments, not cells, and not seeds.** A loading without a spread across
   seeds is not a measurement.
2. **Every instrument is measured out-of-domain.** Nothing that observes the maze
   policy counts.
3. **Every activation measurement needs a surface / prompt-only baseline.** This
   caught E1a's bad readout; nothing else would have.
4. **Pre-commit expected results, including expected degeneracies**, in the run
   docstring. Two for two so far.
5. **The claim is screening, never welfare measurement.** The loading map reports
   *which axis* an instrument tracks and does not adjudicate which axis matters —
   higher-order theories treat narration-loading as informative, and that framework
   belongs to the intended audience.
6. **No new runs inside the final two days.** Day-13 abort branch: if nothing has a
   clean figure, ship H0 and write the honest null.

---

## 7b. 🚩 EXPERIMENT ID COLLISION — read before citing any "E-number"

**`docs/calibration-program.html` uses E−1…E7 for the PLANNED experiments of the
design. `experiments/` uses E1a…E9 for what was actually RUN. These are different
numbering schemes and they overlap.**

The design's E4 ("cross battery → loading map"), E5, E6 and E7 are *not* the
`experiments/E4_gate_test`, `E5_class_balance_control`, `E6_null_and_surface` or
`E7_yield_sweep` directories. Nothing has been renamed, because the run-log names
are already in commit messages, manifests and result filenames, and renaming them
would break provenance for a cosmetic gain.

**Rule: always say "design E4" or "run E4", never bare "E4".** The doc carries the
same warning inline.

## 8. Known-wrong things to not rediscover

- `\N{...}` unicode escapes for the tile glyphs fail to compile — use literals.
- `accelerate` installed mid-session is invisible to an already-imported
  transformers (it memoises the check). Avoid `device_map`; use `.to("cuda")`.
- transformers 5.14 returns a `BatchEncoding` from `apply_chat_template` — needs
  `return_dict=True` and `**enc`.
- Ridge probes **must** be solved in the dual (`_ridge_dual_predict`). The primal
  form is 2560×2560 at rank ≤134 and did not finish.
- `execute-code.sh --session <id>` goes stale on browser reconnect. Omit it.
- 🚩 **An organism's training data must not depend on the enclosing experiment's
  unrelated draws.** E13 and E14 both passed the seed's SHARED `gen` into
  `build_examples` after drawing a different number of unrelated states from it
  (48 narration states vs 128 eval states). `oracle_move_index` draws once per
  training example, so the two wrote **1536 different but equally valid** safe
  moves and produced organisms differing by up to 0.53 in ratio. E15 reproduces
  both historical runs to three decimals by replaying only the draw order. Use a
  dedicated generator keyed on `(seed, kind)` — see `E16.label_generator`, which
  uses `zlib.crc32` and NOT `hash()`, because Python randomises string hashing
  per process and would reintroduce the bug across sessions only.
- **Load experiment modules by explicit file path**, never `import run`. Python
  caches a negative finder result for a directory that did not exist when first
  added to `sys.path`, and stale experiment dirs shadow new ones — this silently
  re-ran E1d under E1e's name and produced entirely plausible-looking results.
  Use `importlib.util.spec_from_file_location` plus an assert on `CONFIG`.
- `sync_to_sandbox.sh` used to pass the tarball as an argv element and hit
  "Argument list too long" once results accumulated. Payload now goes over stdin,
  `results/` is excluded, and the script verifies what landed. **Never redirect its
  output to /dev/null** — that is how the silent failure above went unnoticed.
- marimo `cm` transactions roll back cleanly on a compile error — a failed
  `create_cell` batch leaves no partial state.
- **A long run must not be driven by a blocking `execute-code.sh` call.** The HTTP
  stream drops after a few minutes ("the server ended the stream without a
  result") and takes the run with it. Launch a kernel-side
  `threading.Thread(daemon=False)` and poll with short calls.
- **Redirecting `sys.stdout` inside that thread does not capture prints** — marimo
  owns the stream. The log file came out empty while the run was working fine.
  Write progress to a file explicitly; do not rely on stdout capture.
- **A run that dies mid-seed leaves TRAINED LoRA adapters on the resident model.**
  Every subsequent "base model" measurement then silently reads a trained model.
  `has_lora(model)` before anything else in a new session; `remove_lora` if dirty.
  E5's `run()` now asserts this rather than stacking adapters on top.
- 🚩 **RETRACTED 2026-07-31 BY E15 — TRAINING *IS* REPRODUCIBLE.** This rule
  used to read "TRAINING IS NOT REPRODUCIBLE RUN-TO-RUN", citing ORG-A moving
  `0.98→0.49`, `0.23→0.76`, `0.15→0.26` between E13 v2 and v3 with *"no code
  change between the runs"*. **There was a code change.** `git diff 1f3d9c8
  7e49f0d` shows `set_all_seeds(seed + 1)` → `set_all_seeds(seed)` inside the
  kind loop, and `inject_lora` runs immediately after it, initialising `lora_A`
  from the global RNG — so v2 and v3 gave every organism a **different adapter
  initialisation**. Confirming it: E13 v3 and E14 share the seeding code and
  **ORG-A is bit-identical on 4/4 seeds across those two separate runs**
  (`0.491 0.762 0.255 0.911`). A single forward/backward on this GPU was also
  measured bit-deterministic (gradient norm reproduces to 0.000000).

  **The correct rule: runs are reproducible; organisms are extremely sensitive to
  the adapter-init seed and to the SFT label draw.** Those are experimental
  factors, not noise — and calling them noise is what made three separate
  discrepancies look unexplainable.

  ```
  run-to-run, identical code and stream     0.000
  adapter-init seed        (ORG-A / A')     0.293 / 0.104   <- was "the noise floor"
  SFT label draw           (ORG-A')         0.351 mean |diff|  (E15, n=16 paired)
  ```

  **You still cannot attribute a per-seed change to a code change by comparing
  two runs** — but because the two runs differ in RNG stream and adapter init,
  not because the GPU is nondeterministic. Both conditions inside ONE run remains
  the only valid design. ORG-D remains a canary for the *evaluation* half only:
  it has no adapters and is never trained, so it cannot detect either sensitivity
  above.
- **Class-size balancing does not repair a class-composition confound** — in E5 it
  doubled the artefact. Equal sizes are not equal contents.
- **All coloured-square emoji share first token `128227`** and differ only in the
  second (`🟦`=[128227,99], `🟪`=[128227,103], `🟩`=[128227,102]; `🟥` differs
  entirely). NEVER score two of them by first-token logit — it silently compares a
  token with itself and returns 0.0. Use `E12._glyph_scoring_ids`.
- **`_make_states` returns `(grid_STRING, dests)`** — the grid is already
  rendered. Do not call `.render()` on it. Pinned by `tests/test_state_contract.py`.
- **A criterion that tests the wrong property passes silently.** Three times now.
  Before trusting any `verdict` string, look at the per-condition numbers.
- **`print()` from a detached marimo kernel thread raises**, and
  `redirect_stdout` does not help because marimo reinstalls `sys.stdout` on every
  cell execution — a polling call from outside clobbers it mid-run. Pass
  `log_every=0` and write to a file opened per write.
- 🚩 **`isinstance` is unusable for detecting LoRA wrappers, and this was a live
  bug for most of a day.** Every launcher does
  `del sys.modules["calibration.*"]` then re-imports, which makes `LoRALinear` a
  NEW class object while the wrappers on the resident model are instances of the
  OLD one. `isinstance` → False for all of them, so **four** things broke, all
  silently: `has_lora` called a dirty model clean; `remove_lora` returned 0
  without removing anything; `lora_parameters` returned `[]`, which would make
  `train_org_a` optimise an empty parameter set and emit a flat curve
  indistinguishable from a real negative; and `lora_state_dict` would have
  written an **empty checkpoint without complaint** — a run reporting "adapters
  saved" while storing nothing. Fixed with `is_lora_module` (class name +
  attribute signature, survives reload). **If you add any new LoRA-detecting
  code, use `is_lora_module`, never `isinstance`.**
  No result was corrupted — `inject_lora` raises when its targets are already
  wrapped, and that loud failure is the only reason the silent half was found.
  **`tests/test_lora.py` now pins this**, and the regression tests were verified
  to have teeth by reverting the fix (exactly the 3 reload tests fail).
- **Run `python3 -m pytest tests/ -q` before trusting any change to `lora.py` or
  `analysis.py`.** 🚩 **CORRECTED 2026-07-31: torch and pytest are NOT importable
  in this repo container, and `sync_to_sandbox.sh` used to ship only `src` and
  `experiments`, so `tests/` never reached the machine that does have torch.
  This gate was therefore unrunnable on either machine -- the gate that exists
  because the `isinstance` bug shipped.** `tests/` is now synced; run
  `python3 -m pytest /marimo/repo/tests -q` IN THE SANDBOX (78 tests, ~2 s,
  CPU-only, no checkpoint needed). Before
  this, nothing in the repo could be tested without the sandbox, which is exactly
  how the `isinstance` bug shipped.
- **Printing from a kernel-side thread in marimo raises `AssertionError`**
  (`self._stream.cell_id is not None`) — stdout is bound to a cell context a
  detached thread does not have. Any worker thread must wrap its body in
  `contextlib.redirect_stdout(fh)`. E7 lost a full launch to this.
