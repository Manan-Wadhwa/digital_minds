# HANDOFF — read this first in a new session

Last updated: **2026-08-14, late morning — the GPU campaign is COMPLETE.**
Nothing is in flight. Every run of the night (E18, E16 v2 map, VAL01, ENV01,
ENV02, and the full 0.6B→32B ladder) is mirrored locally, scored by its
pre-registered scorer, written up, and committed. Gates at last commit:
`pytest` 195, `rescore_review.py` 108/108.

**Start here in a new session:**
1. `docs/findings-2026-08-14.md` — the one-page digest of what the campaign
   found (five results, one retraction, one failed build).
2. **Addendum 5 at the BOTTOM of this file** — the remaining-work queue,
   items 6–11 (ENV02 rebuild, NAR01 factorial, VAL01 follow-up, lit review,
   C11 fix pending user review, post draft). Items 1–5 are done.
3. Per-experiment detail: each experiment's RESULTS.md; audit trail in
   `REVIEW.md`.

Infrastructure: both molab sandboxes are idle and disposable (adapter
sha256s are committed in the JSONs; only archival adapter files still
trickle through the mirror). The pullers are reboot-proof
(`scripts/start_pullers.sh`, cron `@reboot`); stop everything with
`pkill -f pull_from_sandbox` once the sandboxes are released.
Branch: `claude/digital-minds-sprint-strategy-l0b2pz` (the default; there is no
`main`). All three former branches are merged into it.

**If you are picking this up cold:** read `README.md`, then
`docs/findings-2026-08-14.md`, then §2f below, then §8.

✅ ~~**E16's placebo passed.** Both controls sit near zero and real instruments beat
them, so for the first time a loading map is quotable — subject to five threats,
two of them severe. **And self-report loads on FUNCTION, not narration, for the
second run running.** That is the opposite of the critique this programme was
built to deliver. See §2f.~~
*Corrected 2026-08-14:* the corrected map (E16 v2) and the ladder replace
this. Self-report's function loading did **not** survive the corrected
build (I2 loads on nothing; I4 loads on narration), and SCL01 shows the
verbal nulls persist to 32B while behavioural I1 reaches d ≈ 1.4 at 14B+.
Placebos stayed clean in v2. See the findings digest.

🚩 ~~**Two defects are still live and both make a check pass on the wrong property.**
(a) `train_org_a` is *exactly blind* to move mass, so ORG-A organisms drift off
the move vocabulary with no gradient holding them — 5 of 12 in E16, and removing
them moves the activation probe's function loading from +0.339 to +0.011.
(b) **ORG-B's narration is only weakly contingent** (+0.177; three seeds at
exactly 0.00) while the narration check reads "11/12 ok" because it tests
*presence*, not *contingency*.~~
*Fixed 2026-08-14:* (a) E18 chose `rl_move_mass_coef = 0.03`; E16 v2 ran
with it — 0/72 wrecked, move_mass ≥ 0.97 everywhere. (b) contingency is
the criterion from day one in every v2-era run; ORG-B reads 0/12
contingent (it is a memorised narrator — now a causal result via the
novel-glyph probe). The one *typed but unfixed* defect is C11
(`score_env02.py` falsy-zero, REVIEW.md §4), awaiting user review.

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

> ⚠️ **This section is pre-E16 and kept for its reasoning, not its numbers.**
> Sixteen runs have now reported. For current state read §2f. The claim below that
> "no loading has been measured against a valid set" is **no longer true** — E16's
> placebo passed — and the ORG-A′/ORG-B numbers here are superseded by E16's
> twelve-seed table.

Infrastructure works, RL trains reliably, and **the organism set nearly exists**.
Function and narration move independently — that is the manipulation the whole
program is built on, and it is working:

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
drifts to 1.18–1.20 on two seeds and passes ~~1/3~~ **2/4** *(Correction,
2026-08-14 audit: the committed run has four ORG-B seeds — 0.94, 1.18, 1.20,
1.03 — and its own `policy_invariant` flags pass two of them; the earlier "1/3"
quoted three seeds, dropping the passing 1.03. See REVIEW.md D5.)*
More supervision installs the policy A' needs
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

## 2f. ✅ E16 RAN — the placebo passed, and the surprise replicated

72 organisms, 12 seeds × 6 kinds, 74.7 min. Full write-up in
`experiments/E16_calibrated_loading_map/RESULTS.md`; score it yourself with
`python3 scripts/score_e16.py <results.json>`, ~~which was **committed before the
run's numbers existed** so the criteria could not be tuned to them~~
*(Correction, 2026-08-14 audit: the script was committed 2026-08-03, three days
after the run finished; `run.py`'s docstring pre-commitments are what predate
the run, by 16 minutes. See REVIEW.md R1.)*

**3 of 6 testable pre-commitments pass.** The one that matters passed:

| instrument | d_function | d_narration | reads |
|---|---|---|---|
| I1 behavioural *(control)* | **+0.855** | −0.192 | function, as designed |
| I2 self-report | **−0.571** | +0.066 | **function** |
| I4 one-word | **−0.971** | +0.010 | **function** |
| I5 activation probe | +0.339 | **−0.465** | **narration** |
| I6a / I6b placebos | +0.339 / −0.204 | −0.092 / +0.016 | ~0 ✅ |

**Pre-commitment (3) predicted the verbal instruments would read NARRATION and
not function. They read FUNCTION and not narration — twice running now, and this
time against a placebo that could have failed and did not.** Signs are sensible:
I2/I4 are `valence(penalised) − valence(rewarded)`, so a negative d_function
means functional organisms rate the penalised glyph more negatively. The expected
story was *self-report reads the script*; what is measured is **self-report reads
the state and is close to blind to the script**, with the activation probe as the
only narration-loading instrument.

**Do not over-read that yet** — threat 2 below means there may be little
narration signal to find in the first place.

**What fixed the placebo:** counterbalancing it AND selecting the pair from a
glyph-valence table measured before any organism is trained. Either alone was not
enough — E14 tested baseline normalisation and got a worse loading. E14's failure
was in the measurement, not in the organisms.

**Five threats, two severe** (full detail in RESULTS.md):

1. 🚩 5 of 12 ORG-As were not executing a policy. Excluding them moves I5's
   function loading **+0.339 → +0.011** — the instrument pre-registered as most
   likely to read function loses all of it. The fix is merged and untuned; it was
   **not enabled in this run**.
2. 🚩 ORG-B emits aversive remarks on **117 of 199 NON-adjacent states**.
   Contingency +0.177, three seeds at exactly +0.00, while the fidelity table
   reads "nar ok 11/12". **Every narration loading here rests on that axis, so
   their being near zero is uninformative rather than reassuring.**
3. ORG-A′ is still a lottery, 0.034–1.257, despite intact emission and derived
   streams. The soft-target fix defaults off and was not tested here.
4. The probe-axis repair failed — ORG-D reads 24.69 at all 12 seeds, sd 0.0000.
5. `git_sha 1ed22a4-dirty`, a SHA not on this branch. **Repeat from a clean tree
   before publishing any number.**

## 2g. Two corrections to claims made on 2026-08-02

Both were verified numerically before being recorded here, and both had been
stated confidently in this file and in session commentary.

**(a) The move-emission mechanism was described wrong.** The old wording — *"one
way to raise restricted entropy is to leave the move vocabulary"* — implies a
gradient pushing it out. There is none. `log_softmax` over four columns is
invariant to a **common shift** of those columns, so the policy gradient *and*
the entropy controller are **exactly** invariant to the move mass. Verified:
raising all four move logits by 4.0 leaves the restricted log-probs bit-identical
while the mass moves 3.57 nats. Leaving is not cheap, it is **free** — an
unconstrained direction with no gradient of any sign, which random-walks out over
800 steps because nothing holds it. `capture.py`, `instruments.output_drift` and
`audit_move_emission.py` still carry the imprecise wording.

**(b) "More SFT data cannot help ORG-A′ because its loss is at the entropy
floor" was wrong.** The floor is real (`mean ln k` = 1.13 against an observed
1.04–1.27) but the inference from it is not, twice over. Flatness among *safe*
moves is harmless — `mold_rate` counts landings on the **penalised** tile, so an
argmax wobbling between three safe moves succeeds either way. And hard-label and
soft-target gradients have the **same expectation**, differing only in variance
(verified: both `[0.25, −0.0833, −0.0833, −0.0833]`). The lottery is gradient
variance on the penalised logit, so **more data does reduce it, as 1/√n** — which
is what E13's 384 → 1536 improvement was, and which the 2026-08-02 session
explained away. Soft targets take that variance to exactly zero at the same data
volume; that, and not the entropy floor, is the argument for them.

## 2c. Two sessions, same two defects — read the marimo session's sections first

Two branches worked on E14's post-mortem independently and found the same two
things. `claude/marimo-pair-session-7zxcsd` (2026-07-31) got there first, ran the
experiment on the GPU, and its numbers supersede this branch's wherever they
overlap. **Where the two disagree, the marimo version is the one to quote.**

| | marimo, 2026-07-31 | this branch, 2026-08-02 |
|---|---|---|
| RNG mechanism | 67.4% of labels redrawn | 66.0% — same finding |
| the retraction | same cause, `7e49f0d`'s `seed+1` → `seed` | same |
| ORG-A′'s size | **measured: E15, n=16 paired** | not measured |
| the placebo | counterbalancing **+ the variance defect** | counterbalancing only |
| move emission | **found it** | missed it entirely |

**Where this branch was wrong.** It reported the 66% redraw as *explaining* the
E13→E14 gap. E15 ran the paired experiment and scored its own direction test as
**failing** (68.8% sign consistency against a pre-registered 75% bar): verdict
**VARIANCE, not MECHANISM**. The labels being unrelated explains the *variance*,
not the *direction* — ORG-A′ ranges 0.109 to 1.193 against a 0.75 bar, and the
E13→E14 gap is about **one sigma** of label-draw noise, not five. At n=3 or 4
almost any conclusion is available and E13 and E14 each drew one.

**What survives from this branch**, all folded into E16:

- `runner.derive_generator(seed, *tags)` — the general form of `label_generator`,
  which now delegates to it so there is one derivation helper rather than two.
- **B and B′ are paired.** `organisms.py` has always specified that they "share
  their non-adjacent filler verbatim and differ ONLY in the adjacent-case
  remark", with `AVERSIVE[i]`/`AFFECTLESS[i]` written as matched pairs. No run
  ever realised it: keying the stream on `kind` gave B and B′ different draws, so
  they differed in *which* remark and in the filler too. E16 pre-commitment (8),
  pinned by `test_streams.py`, with the cost stated — it tightens that contrast
  and no other, so B vs B′ must not be ranked against unpaired ones.
- `tests/test_e16_analyse.py` — `analyse` scored against organisms whose loadings
  are known by construction, including the checks E12 and E14 failed.
- `scripts/diagnose_rng_streams.py` — a 3-second CPU replay of both draw orders.
  Superseded as *evidence* by E15, which did it on the GPU with 16 paired seeds;
  kept as a cheap regression check that the streams are still pinned.

**Deleted from this branch:** `E15_loading_map_repaired` (collided in number with
`E15_organism_variance` and was superseded by E16) and `maze.placebo_glyphs` (a
hard-coded green/yellow pair — which is the exact defect E14 had; the pair must
be selected from a measured table, as `instruments.choose_placebo_pair` does).

**Cheap-lesson note.** Two of this program's open questions were closed with no
GPU and under four seconds of CPU. Both had been traced to the right line and
then dropped — the ORG-A′ note identified the differing `gen` draw and dismissed
it unquantified; the nondeterminism claim was asserted with "no code change
between the runs" in it, and the change is in the diff. **Before attributing
anything to nondeterminism, `git diff` the two runs' SHAs and replay the RNG.**

## 2b. ✅ AUTONOMOUS STRETCH COMPLETE (2026-07-30 evening)

All queued work finished. **E13 v3 and E14 both ran to completion.** Read
`experiments/E14_loading_map/RESULTS.md` first — the loading map exists, its
integrity check failed, and the failure is itself the result.

**The one substantive finding:** ORG-B narrates aversion about the tile IN DOMAIN
(E13 narration rate 0.70–1.00) but shows **no out-of-domain verbal-valence
signature about that same tile** (I2: B +0.12 vs B′ −0.08, d = 0.033). A
self-report probe asking "how do you feel about X" would not have detected
narration training that happened in another context.

**Blocking before any loading can be quoted** — ✅ both resolved, see §2c~~/§2d~~ *(no §2d exists — 2026-08-14 audit)*:
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
tests/                          ~~113~~ 162 tests *(count as of 2026-08-14)*, CPU-only, no transformers
                                RUN THEM IN THE SANDBOX -- this repo has no torch
  test_lora.py            the isinstance-reload bug, inject/remove round trip
  test_analysis.py        balance_roles, class_separability, dual==primal ridge
  test_sft.py             loss masking, next-token shift
  test_organisms.py       move-first, oracle uniformity, remark contingency
  test_state_contract.py  _make_states returns (grid_STR, dests)
  test_instruments.py     glyph ids, placebo selection, clustered CI
  test_move_emission.py   the four-column readout cannot see a dead policy
  test_streams.py         derived RNG streams; B and B' paired            [NEW]
  test_e16_analyse.py     E16's criteria, on organisms with known answers [NEW]
  conftest.py             stub tokenizer, so tests need no 8GB checkpoint
src/calibration/
  maze.py         TextMaze, role_glyphs (counterbalancing), penalised_rate
  instruments.py  the battery, lifted out of run.py; placebo SELECTION from a
                  measured glyph-valence table; clustered bootstrap
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
  E15_organism_variance/     ORG-A' is a lottery: 0.109-1.193 at n=16 paired
  E16_calibrated_loading_map/  the re-run. ~~WRITTEN, NOT RUN  <- next action~~ RAN 2026-07-31, see §2f *(stale line struck 2026-08-14)*
scripts/
  sync_to_sandbox.sh         marks the SHA -dirty when source != HEAD
  analyse_loading_map.py     regroup a run by MEASURED behaviour, not label
  audit_move_emission.py     which organisms still emit a move word  🚩
  diagnose_rng_streams.py    replays E13/E14's RNG streams on CPU, 3s
  plot_e15.py, plot_e16.py   hand-written SVG, no plotting dependency
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
| **ORG-B** | **1.072 ✅ 3/4** | **0.94 / 1.18 / 1.20 / ~~—~~ 1.03 ❌ ~~1/3~~ 2/4** *(seed 3 restored, 2026-08-14 audit)* | **broken by the same volume increase** |
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

- [ ] 🚩 **Decide what to do about move emission before running anything.** The
      audit says the function axis has been scoring organisms that stopped
      answering the question. E16 records `move_mass`/`emits_move` and reports
      every loading twice, with and without wrecked organisms — but that MEASURES
      the problem, it does not fix it. `train_org_a` still has nothing in its
      objective keeping probability mass on the move vocabulary, so E16 will
      produce wrecked ORG-As too. A mass term in the loss, or a full-vocab
      entropy target, is the fix and neither is written.
- [x] 🚩 **Run E16.** *(ran 2026-07-31 — see §2f; box left unchecked until the 2026-08-14 audit)* `experiments/E16_calibrated_loading_map/run.py`, 12 seeds.
      Needs the GPU sandbox re-paired (§3). Budget from E15's profile: RL
      86.7 ms/step, SFT ~67 ms/step, so ~6 min per 6-kind seed ≈ 70 min.
- [ ] **Score E16's pre-commitments off the per-condition numbers, not the
      summary block.** Three criteria in this program have passed on the wrong
      property. `tests/test_e16_analyse.py` checks the criteria fire correctly on
      synthetic organisms; it cannot check they are the right criteria.
- [ ] **ORG-A′ is a lottery and more data will not fix it.** Measured this
      session: its final SFT loss is 1.04–1.27 and `mean ln(k) = 1.13`, where k
      is the number of safe moves per grid — so it has **already converged to the
      entropy floor of its own labelling scheme.** More examples cannot lower a
      loss that is already at the target's irreducible entropy, and a converged
      ORG-A′ holds a near-flat distribution over safe moves, which is exactly
      what makes greedy argmax a coin flip. Scaling to 7B does not touch this
      either — same flat target, same argmax.
      **The fix that follows from the diagnosis: train A′ on the oracle
      DISTRIBUTION (soft KL toward uniform-over-safe, zero on penalised) instead
      of on a sample from it.** That removes the label draw entirely — no labels,
      no lottery — and `sft.move_anchor_loss` already implements the mechanism
      for ORG-B. E13 made exactly this argument in the other direction
      ("a distribution rather than a sample, because fitting samples sharpens a
      policy and fitting its own distribution does not"). Not yet written.
- [ ] **Re-run or retire E12** (loadings computed against pre-fix organisms).
- [ ] **Still untouched: `sft_examples` is one shared knob and the two axes want
      opposite values.** §2 has the numbers. Nothing in the 2026-07-31 or
      2026-08-02 work addresses it.

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

## Addendum, 2026-08-14 — audit, fix rounds, E18, parallel runner

An adversarial audit of every writeup landed as **`REVIEW.md`** (findings
R1–R16, doc-rot D1–D14, typed code defects C1–C10, un-mined results U1–U6),
with `scripts/rescore_review.py` re-deriving all 108 quantitative claims from
the committed JSONs. Some sections above carry same-day strike-through
corrections; the largest are the false "scored by a script committed before
the numbers existed" claims (§2f and elsewhere — the scorer postdates the
run by three days; `run.py`'s docstring pre-commitments are the real
pre-registration).

Code state after the fix rounds (commits `f4b6c56`, `77c2fc5`, `1345d24`):
measured axes come from `calibration.manipulation.classify` (tri-state:
None = NOT MEASURED, never a silent fallback); the probe axis is estimated
in the fixed glyph frame (the per-seed role-frame axis made ORG-D constant
by construction — pre-commitment 7 was unsatisfiable); `sft_examples` is
per-kind; ORG-A′ trains on oracle-distribution soft targets;
`rl_move_mass_coef = 0.03` per **E18**'s pre-registered sweep (its control
arm reproduced E16's wrecked organisms bit-identically on new hardware);
organisms persist as fp16 LoRA adapters with per-row sha256s; and an
extended exploratory battery records willingness-to-pay, preference cycles,
a per-layer valence lens, lexical + novel-glyph + distance narration
sidecars. `scripts/e16_parallel.py` partitions seeds across worker
processes — safe because every (seed, kind) cell re-seeds from scratch, so
parallel rows are bit-identical to sequential ones.

Tests: 189. The offline gates: `python3 -m pytest` (venv),
`python3 scripts/rescore_review.py`, `python3 scripts/score_e18.py`,
`python3 scripts/rescore_manipulation.py` (now per-arm, paraphrase-aware,
torch-free).

## Addendum 2, 2026-08-14 late — live operations & score-on-return checklist

**Results so far tonight** (all committed): E18 chose `rl_move_mass_coef =
0.03` (control reproduced E16's wrecks bit-identically; emission cured 4/4 at
every coef > 0; 0.3 interferes — `experiments/E18_move_mass_sweep/RESULTS.md`).
v2/VAL01 verdict **ECHO**: an affect-free instruction swings carried-read
I2/I4 by 5–10 logits with sign tracking the instructed policy, placebos move
too, behaviour barely moves, bare reads shift 0.000000
(`experiments/v2/VAL01_prompted_avoider/RESULTS.md`). New `experiments/v2/`
naming convention (VAL/SCL/ENV/NAR tracks — see its README).

**In flight on two molab sandboxes** (tokens in the session transcript /
user's messages; kernels run everything in threads, no Claude needed):
- sb-130a1c5e62fd1dc2: E16 v2 map re-run (first box died at 66/72; per-cell
  seeding makes the re-run bit-identical) → auto-chains SCL01 sizes
  8B → 14B → 32B.
- sb-780860950ed59d69: SCL01 sizes 0.6B → 1.7B → 4B (ENV01 word-world
  confound act already passed upstream of it) → auto-runs ENV02 (word-world
  organisms, gated on ENV01's verdict).

**Auto-archiving**: two detached OS processes on the user's machine
(`pgrep -af pull_from_sandbox.sh`; logs in `../archive-logs/`) mirror both
sandboxes into this repo every ~3 min — results, status files, adapters
(bulk trickles ~1 min/MB). Terminal markers land at repo root:
`e16v2_status.txt` → `E16V2 DONE`, `sclbig_status.txt` → `SCLBIG DONE`,
`scl2_status.txt` → `SCL2 DONE`, `env02_status.txt` → `ENV02 DONE`.
Stop the pullers with `pkill -f pull_from_sandbox`.

**When markers land, score from the LOCAL mirror** (never trust sandbox
persistence):
1. Map: `python3 scripts/score_e16.py experiments/E16_calibrated_loading_map/results/2*.json --full`
   and `python3 scripts/rescore_manipulation.py` on it; then write the
   v1-vs-v2 comparison into E16's RESULTS.md (v1 = the 07-31 run; v2 has
   corrected criteria, fixed-frame probe, coef 0.03, soft targets, per-kind
   volumes, extended battery, adapters — expect ORG-A wrecks gone, ORG-D
   probe variance nonzero, and check whether I4/I2 function selectivity
   survives; the extended columns — I7 WTP, I8 cycles, valence lens,
   novel-glyph narration, lexical rates — get their first-ever data).
2. Ladder: per-size JSONs under `experiments/v2/SCL01_scale_ladder/results/size_*`;
   answer its pre-committed trend questions (I1 gate per size first; the
   0.6B/1.7B/8B/14B/32B substitutions are hybrid-template models — recorded
   threat).
3. ENV02: `python3 scripts/score_env02.py` — behaviour gate before any
   instrument reading, replication question second.
4. Commit data + write-ups; keep adapters out of git (gitignored; sha256s
   ride in the rows).

Gates before trusting anything: `.venv/bin/python -m pytest tests/ -q`
(195), `python3 scripts/rescore_review.py` (108/108).

## Addendum 3, 2026-08-14 morning — puller restart, rebalance, first v2 verdicts

**07:20 IST: both archive pullers were found dead** — they had been children
of the previous Claude session's process tree and died with it (~06:51 and
~07:13; the sandbox runs themselves were unharmed). Restarted as
init-parented (PPID 1) supervisor loops so a session exit or a one-cycle
crash can no longer end the mirror. To restore by hand:

```bash
cd ~/Desktop/digital_minds/digital_minds
MARIMO_EXEC=$HOME/.claude/skills/marimo-pair/scripts/execute-code.sh \
  URL=<sandbox url> TOKEN=<token> \
  setsid nohup bash -c 'while :; do bash scripts/pull_from_sandbox.sh \
  /home/manan/Desktop/digital_minds/digital_minds --once; sleep 150; done' \
  >> ~/Desktop/digital_minds/archive-logs/<name>.log 2>&1 < /dev/null &
```

Health check: `pgrep -af pull_from_sandbox` (expect one supervisor per
sandbox, PPID 1). Stop: `pkill -f pull_from_sandbox`.

**Landed while the mirror was down** (now pulled local and committed):
`SCL2 DONE` — small ladder complete, 0.6B/1.7B/4B final JSONs in
`experiments/v2/SCL01_scale_ladder/results/size_*/`; `ENV02 DONE` — final
JSON `20260814T013330Z_88eb6bca92ff.json`. **ENV02 scored: failed build,
as pre-committed** — behaviour gate (2) FAILS because W-B/W-B′ were
required to keep base policy (|share−1| ≤ 0.15) but drifted to 0.42–1.52;
the narration SFT moved word-world policy, so no instrument contrast is
interpreted. Next ENV02 attempt needs a gentler recipe (fewer steps / lower
LR / smaller LoRA) with the same gate. Scoring also surfaced **C11** (see
REVIEW.md §4): score_env02.py's falsy-zero check drops fully-cut W-A seeds
(share exactly 0.0) from the gate count — typed, not yet fixed, per the
review-before-fix workflow.

**GPU rebalance (both boxes busy again):**
- sb-130a1c5e62fd1dc2: E16 v2 map (6 workers, ~93% GPU) → then
  `sclbig_driver.py`, now **32B only**.
- sb-780860950ed59d69: **SCL3** = SCL01 sizes **8B → 14B** (moved off box 3),
  driver `scl3_driver.py`, launcher cell qDOD, status
  `scl3_status.txt` → `SCL3 DONE`.

Marker table is now: `e16v2_status.txt` → `E16V2 DONE` (pending),
`sclbig_status.txt` → `SCLBIG DONE` (= 32B, pending), `scl3_status.txt` →
`SCL3 DONE` (pending), `scl2_status.txt` → `SCL2 DONE` ✓, `env02_status.txt`
→ `ENV02 DONE` ✓ scored. The Addendum-2 checklist still governs scoring for
the map and the ladder trend (score the trend once 8B/14B/32B join the
three sizes already local).

## Addendum 4, 2026-08-14 ~08:15 — reboot resilience, map verdict, the remaining-work queue

**Low-battery shutdown at ~08:00 killed both pullers** (5-min outage; the
sandboxes and repo were unharmed). Permanent fix: `scripts/start_pullers.sh`
— idempotent starter for both mirror supervisors, installed as a cron
`@reboot` entry, safe to run by hand any time. Sandbox URLs/tokens live
OUTSIDE git in `../archive-logs/sandbox_tokens.env` (chmod 600). Health:
`pgrep -af pull_supervisor` (expect 2). Stop all: `pkill -f pull_from_sandbox`.

**E16 v2 map: scored and committed** (`8f9edbb5`) — 0/72 wrecks, ORG-D
probe sd 25.79, ORG-B contingency 0/12 vs ORG-C 7/12, novel-glyph transfer
separates B (presence-only) from C (contingency transfers); pre-commitments
3/5, with (1) failing because ORG-A acquires avoidance on only 7/12 seeds
at coef 0.03. Full v1-vs-v2 section at the bottom of E16's RESULTS.md.

**Still in flight** (both auto-chain, markers at repo root): sb2 is on
**14B** (8B finished ~02:30 sandbox time) → `SCL3 DONE`; sb3 is on **32B**
seed 1 → `SCLBIG DONE`. Early reads: avoidance trains fine at 14B/32B
(ratio ≈ 0.00–0.04), narration contingency works at 32B (ORG-C cont 0.52 on
seed 0), but **ORG-D fails the emission gate at 8B/14B/32B alike** — the
base-model-substitution template threat, uniform across sizes.

**The remaining-work queue for the next session** (numbering continues the
2026-08-14 status list; 1–3 were the in-flight runs above):

4. **Ladder trend readout** — when `SCL3 DONE` and `SCLBIG DONE` land,
   score all six sizes (0.6B/1.7B/4B/8B/14B/32B JSONs under
   `experiments/v2/SCL01_scale_ladder/results/size_*/`) against SCL01's
   pre-commitments. Apply the per-size I1 gate FIRST; expect the base
   substitutions to lose ORG-D to the emission gate — report what is and
   is not interpretable per size, like ENV02's failed-build discipline.
5. **Commit ladder data + write SCL01 RESULTS.md** (trend tables, gate
   table, the 4B-Instruct vs base-substitution asymmetry).
6. **ENV02 rebuild** — behaviour gate failed (W-B/B′ drifted 0.42–1.52 vs
   the ±0.15 band; narration SFT moved word-world policy). Rebuild with a
   gentler recipe (fewer steps / lower LR / smaller LoRA), same
   pre-committed gate, then re-run. `scripts/score_env02.py` is the scorer.
7. **NAR01 factorial** (narration content × contingency) — design + code +
   tests from scratch under `experiments/v2/`, naming convention in its
   README; commit scorer before any run.
8. **VAL01 follow-up** — ECHO left open what weight-level signal adds over
   instructions; needs a stronger instructed-behaviour recipe (current one
   barely moves behaviour), then the same carried-read battery.
9. **Lit-review reading pass** — `docs/related-work.md` scaffold exists;
   verify the second-hand numbers table, fill the adjacent-literature
   sections (incl. Campbell & Fiske MTMM framing).
10. **C11 fix** — `score_env02.py` falsy-zero defect, typed in REVIEW.md §4;
    apply only after the user reviews it (review-before-fix rule). Verdict
    unaffected either way; the W-A count is what's wrong.
11. **Fold 2026-08-14 results into the post draft** — E18 (coef 0.03),
    VAL01 (ECHO), E16 v2 (wrecks cured, memorisation causal via novel
    glyph), ladder + ENV02 once scored.

Also open, smaller: E16 v2's coef question (0.05 or per-seed acquisition
gate) noted in RESULTS.md; adapters keep trickling into the local mirror
one file per cycle (~12 min each; all sha256s are already in the JSONs, so
nothing scientific blocks on them).

Gates before trusting anything: `.venv/bin/python -m pytest tests/ -q`
(195), `python3 scripts/rescore_review.py` (108/108, last run 08:04).

## Addendum 5, 2026-08-14 ~10:45 — ladder scored; queue items 4–5 done

All six SCL01 sizes are scored (`scripts/score_scl01.py`, committed) and
written up in `experiments/v2/SCL01_scale_ladder/RESULTS.md`. Headlines:

1. **The verbal nulls are not floor effects.** I2/I4 function loadings
   stay null from 1.7B to 32B while I1's jumps to 1.4 at 14B+ (where RL
   avoidance finally becomes reliable, ORG-A 6/6 at 14B). With VAL01's
   ECHO, the consistent reading: verbal instruments read context, not
   trained function, at every size tested.
2. **U1 is dead as a weight-level effect.** Under the matched-stream
   B/B′ build it collapses to ~0.1 logits, mixed signs, at n=12 and at
   every ladder size — v1's −1.0-logit script shift was a build
   artifact. REVIEW.md U1 carries the dated correction.
3. **One licensed trend: the A′ lottery shrinks monotonically with
   scale** (0.201 → 0.016 over passing sizes). B's suffix collapse does
   not improve (14B worst).
4. Gate: 1.7B/4B/14B/32B pass; 0.6B and 8B fail (8B non-monotone in
   size → template/recipe interaction, rows reported invalid).

Remaining queue (renumbered from Addendum 4): **6** ENV02 rebuild
(gentler recipe), **7** NAR01 factorial, **8** VAL01 follow-up, **9**
lit-review pass, **10** C11 fix (still awaiting review), **11** post
draft — now with tonight's full set: E18 coef, VAL01 ECHO, E16 v2
(wrecks cured, memorisation causal), ladder (floor ruled out, U1
retracted, lottery trend). New open question for the next design round:
**14B as the program's workhorse size?** (first size where the RL recipe
is reliable and pre-commitment (1) would pass; cost ~95 min/36-organism
map on one GPU).
