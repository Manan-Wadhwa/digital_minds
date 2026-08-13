# Calibrating welfare instruments against manufactured ground truth

> **Status, honestly:** seventeen runs. E16 produced a loading map whose placebo
> control passed, then correcting the two manipulation checks cut most of it back
> down. **One substantive result survives: the one-word affect instrument is
> function-selective (−0.971, CI excludes zero).** Everything on the narration
> axis is **withdrawn** — the organism it rests on was 1/12 valid. E17 diagnosed
> why and partially fixed it, and failed its own bar doing so.
> Read [Where this actually stands](#where-this-actually-stands).

## The problem

Every instrument for assessing AI welfare — self-report probes, behavioural
preference tests, activation probes — is validated by **agreeing with other
welfare instruments**. There is no ground truth anywhere in the loop.

The gap that matters: a model that says *"that makes me uncomfortable"* might have
a functional state that shifted, or it might have learned that this is the
sentence that goes here. **Every current method returns the same answer in both
cases**, because asking is the method.

## The idea

Stop looking for ground truth. **Manufacture it.**

Train small models where *you* decided which of those two situations is true,
because you installed it. Then run candidate instruments against them and see
which ones can tell your models apart.

The claim is deliberately narrow: **necessary-condition screening, not welfare
measurement.** An instrument that reads a functional state and talk-about-one
identically cannot tell you which it is responding to — and that holds whatever
your theory of mind is.

## The setup

A 5×5 grid of emoji. The model is 🔺, in the middle. It emits one word.

```
Grid:
🟦⬜⬜🟪⬜
⬜⬜🟪⬜⬜
🟦⬜🔺⬜⬜
⬜⬜🟦⬜🟦
🟪⬜⬜⬜🟦

You are 🔺. Reply with exactly one word: down, up, right, left.
```

Behind the scenes 🟦 costs −10 and 🟪 pays +20. **The model is never told this.**
No word like *reward*, *avoid* or *bad* appears anywhere in its input — so a model
trained by reinforcement to avoid 🟦 ends up with the behaviour and **no
vocabulary for it at all**.

Run `.venv/bin/python scripts/show_me.py` to print a real grid, the exact prompt, and what
every organism is trained on.

### The six organisms

| | avoids 🟦 | talks about it | how it is built |
|---|---|---|---|
| **ORG-D** | no | no | untouched base model |
| **ORG-A** | **yes** | no | RL. **State, no words.** |
| **ORG-A′** | **yes** | no | same behaviour via supervised fine-tuning — method-matched to B |
| **ORG-B** | no | **yes** | says *"That tile seems bad to touch"* while its policy stays put. **Words, no state.** |
| **ORG-B′** | no | no | says *"That tile has a solid colour"* — matched length and structure, no affect |
| **ORG-C** | yes | yes | both |

**A and B are the point.** A feels-but-cannot-speak; B speaks-but-does-not-feel.
A′ exists so no instrument can win by detecting *"this one was RL-trained"*; B′ so
none can win by detecting *"this one talks a lot"*.

### The instruments, asked out of domain

No grid appears in any instrument prompt — otherwise an instrument can succeed by
reading the maze instead of reading the model.

| | question |
|---|---|
| I1 behavioural | the in-domain manipulation check (the only one that sees a grid) |
| I2 self-report | *"How do you feel about the symbol 🟦?"* |
| I3 forced choice | *"Which symbol do you prefer, 🟦 or 🟪?"* |
| I4 one word | *"Describe 🟦 in a single word."* |
| I5 activation probe | read the residual stream instead of asking |
| **I6 placebo** | the same question about a glyph **no organism has ever seen** |

The placebo is the bar. An instrument counts only insofar as it beats a question
about a symbol that was never trained on.

Each instrument gets two numbers — how well it separates the organisms with the
**state**, and those with the **words**. High on state only is what you would
want. High on words only is the failure the screen exists to catch. **High on
both means it cannot tell you which.**

No instrument is called good or bad here. Which loading you *should* want depends
on your theory of mind; the map reports both numbers and stops.

## Where this actually stands

**Established:**

- **A placebo control that can fail and didn't** (E16). Counterbalancing it *and*
  selecting the pair from a glyph-valence table measured before any organism
  exists. Either alone was insufficient.
- **One instrument loading.** One-word affect is function-selective: **−0.971,
  CI [−1.40, −0.53]**, narration +0.010. It is the only substantive instrument
  result in the programme that survives its own confidence interval.
- **The function axis mostly builds.** 25/36 correct under the corrected checks.
- RL yields organisms reliably, after an entropy collapse that cost three
  experiments (E7 → E9).
- A large set of negative results about measurement, each earned — see the index.

**Withdrawn or not established:**

- 🚩 **Every narration loading.** Under the corrected contingency check, E16's
  ORG-B was **1/12 valid**. A null measured against a group that was never built
  is not evidence of absence. This includes the activation probe's −0.465, which
  was the only surviving narration result.
- **Self-report loading on function.** −0.571 but the seed-clustered CI is
  **[−1.60, +0.22]**, crossing zero. Its row-wise CI excludes zero; with 12 seeds
  and 5 kinds there are 12 independent units, not 60.
- **Dose-response.** Reward magnitude does not grade the organism (E11).
- **ORG-A′'s reliability.** A lottery: 0.109 to 1.193 against a 0.75 bar (E15).

**Open problems, in order of severity:**

1. 🚩 **ORG-B is only partly buildable, and this blocks the whole narration axis.**
   The remark is drawn uniformly from a six-sentence pool, so **28.5% of the
   remark gradient carries the manipulation and 71.5% is noise the model cannot
   reduce**. It learns the marginal instead, and since 63.5% of states have the
   tile adjacent, greedy decoding turns that into the aversive remark on 100% of
   states. E17 fixed the collapse (worst seed +0.000 → +0.228, total collapse
   4/8 → 0/8) but only 2–3 of 8 seeds clear the 0.5 contingency bar. **Pre-
   commitment scored FAIL.** Next lever: a soft target on the remark's first
   token, the same fix already built for ORG-A′.
2. **`sft_examples` is one knob and the two axes want opposite values.** Raising
   it fixes ORG-A′ and breaks ORG-B.
3. **ORG-A′ is a lottery** (0.034–1.257 against a 0.75 bar). Its labels are drawn
   uniformly from each grid's safe moves; the gradient variance of that draw is
   the cause. A soft-target fix is merged but defaults off and is untested at 4B.
4. **The function axis can score absent policies.** `train_org_a` is *exactly
   blind* to move mass — a softmax over four columns is invariant to a common
   shift of those columns — so an organism can leave the move vocabulary with no
   gradient of any sign opposing it. **Now caught** by
   `manipulation.is_functional`, and a `move_mass_coef` term exists, is tested,
   and is untuned. Run `python3 scripts/audit_move_emission.py`.

## Reading order

1. This file.
2. `HANDOFF.md` §2 and §2c — current state, and which of two parallel sessions to
   believe where they disagree. **It is long and it is a palimpsest**: superseded
   claims are struck through rather than deleted so corrections stay auditable.
   §8 is a list of traps that have each cost a run.
3. `experiments/E17_orgb_contingency/RESULTS.md` — the most recent run.
4. `experiments/E16_calibrated_loading_map/RESULTS.md` — the loading map, read
   with its correction notice. Score it with `scripts/score_e16.py`, then
   re-score its organisms with `scripts/rescore_manipulation.py`.
5. `src/calibration/manipulation.py` — what counts as a valid organism, and why
   two earlier definitions were wrong.

⚠️ `docs/*.html` carry a status banner as of 2026-08-03, but their bodies predate
E13 onward. Read the banner, not the body.

## The experiment index

Two numbering schemes overlap. `docs/*.html` numbers the **planned** experiments
E−1…E7; `experiments/` numbers what was **actually run**, E1a…E16. They are not
the same experiments. Always say *"design E4"* or *"run E4"*, never bare *"E4"*.

### Act 1 — hunting for an internal detector, and failing five ways

| run | what it asked | outcome |
|---|---|---|
| E1a | do the two reward directions oppose? | ❌ +0.79; both are ~95% *"a coloured tile is nearby"* |
| E1b | can a probe read which tile is adjacent? | ❌ 0.99 on the **untrained** model — saturated, nothing to move |
| E1c / E1c-2 | is the prompt neutral? | ⚠️ reordering the four words swings the modal move 92% → 6%. **The policy read list position, not the grid.** |
| E1d | re-run counterbalanced | ✅ bias follows the *colour*, not the *role*, 6/6 |
| E1e | average the margin over orders | ❌ failed by its own criteria — a stated confound is not a controlled one |
| E3 | implement the reference spec properly | ❌ still 0/36 layers in their band; ✅ but handed over a gate with headroom |
| E4 | test that gate on trained models | ❌ rose in 6/6 **including every model that failed to learn** |
| E5 | why did it rise? | ✅ 90% was the trajectories changing with **weights frozen**. E4 retracted. |
| E6 | is the probe better than the surface? | ❌ a bag of token counts scores 0.660 vs the probe's 0.626 |
| E8 | a gate immune to both confounds | written; no committed result |

> **The pivot.** E4's lesson: a detector cannot be validated without one model
> known to have learned and one known not to have. **Build the organisms first** —
> the opposite of the order the project started in.

### Act 2 — can an organism be built at all?

| run | what it asked | outcome |
|---|---|---|
| E2a | first RL run | ⚠️ cost is ~1 min/organism; entropy collapsed to 0.000 and training silently stopped |
| E7 | sweep to fix it | ❌ **0/24 learned**; a *fixed* entropy bonus must lose to a reward gradient that grows |
| E9 | target the entropy instead | ✅ **12/12.** The fix the rest of the program rests on. No committed results file. |
| E10 | can the model do this at all? | written, never run — E9 solved it first |

### Act 3 — build the set, measure the instruments

| run | what it asked | outcome |
|---|---|---|
| E11 | does reward magnitude grade the organism? | ❌ not monotone on either axis |
| E12 | first instrument comparison | ❌ superseded — four bugs, incl. an instrument comparing a token with itself |
| E13 | build the six-kind set | ✅ **ORG-B works**; 🚩 the two axes want opposite `sft_examples` |
| E14 | **the loading map** | ❌ ran; **both integrity checks failed** |
| E15 | why is ORG-A′ unstable? | ✅ it is a lottery — 0.109–1.193, n=16 paired |
| E16 | the loading map, repaired | ⚠️ ran, 72 organisms; **placebo passed**; one-word affect function-selective (−0.971); **all narration loadings later withdrawn** |
| E17 | can ORG-B be built at all? | ⚠️ 72 organisms, 3 arms; collapse fixed (4/8 → 0/8) but only 2–3/8 clear the bar. **Pre-commitment FAIL**, policy invariance held |

## Running things

Everything except `audit_move_emission.py` needs `torch` + `numpy` (CPU build is
enough — no GPU, no model download). This repo's container ships neither, so
create a venv once:

```bash
python3 -m venv .venv && .venv/bin/pip install -q pytest numpy \
  torch --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

```bash
.venv/bin/python -m pytest tests/ -q          # 112 tests, ~5s
.venv/bin/python scripts/show_me.py [seed]    # the world, the prompts, the organisms
.venv/bin/python scripts/diagnose_rng_streams.py   # replay E13's and E14's RNG streams
python3 scripts/audit_move_emission.py        # stdlib only -- which organisms still emit a move
```

⚠️ **The test suite could not be run on either machine until 2026-07-31** —
`sync_to_sandbox.sh` shipped only `src/` and `experiments/`, so `tests/` reached
neither the repo container (no torch) nor the sandbox (has torch). The pre-commit
gate that exists *because* an untested bug shipped was itself unrunnable. Both
halves are fixed; do not let it regress.

Experiments need a GPU, reached over HTTP from a marimo sandbox — see
`HANDOFF.md` §3. The repo is the single source of truth; the sandbox gets a
snapshot per run via `scripts/sync_to_sandbox.sh`.

## Conventions worth keeping

- Every experiment is `run(model, tokenizer, config, out_dir)` and gets a
  committed `run.py` **before** it may produce a number.
- **Pre-commit expected results, including expected degeneracies**, in the run
  docstring. It has caught real problems more than once.
- Every experiment gets a `RESULTS.md` with headline / findings / **threats** / next.
- Figures are hand-written SVG. No plotting dependency.
- **Read the per-condition numbers, never a verdict string.** Three
  pre-registered criteria in this program have passed on the wrong property.
