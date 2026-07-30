# HANDOFF — read this first in a new session

Last updated: **2026-07-30**, after E4 (gate test — the gate failed).
Branch: `claude/digital-minds-sprint-strategy-l0b2pz` · everything below is committed and pushed.

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

Infrastructure works end to end and RL trains (0.111 s/step, ~1 min/organism;
the full design is ~30 min of GPU, so compute was never the constraint). Nine
experiments have run. **Four gate candidates were pre-registered and all four
failed**, and E5 then showed that E4's apparent gate signal was an artefact of its
own measurement protocol — so the program currently has **no validated internal
dose estimator**, only the behavioural one.

**The single blocking problem is organism yield: 1/6 in E5, 2/6 in E4.** No gate
can be validated without organisms that reliably learn, and the representation
effect that E5 makes measurable is currently averaged over five non-learners.
Fix yield with held-out tuning, then re-run E5's decomposition. ORG-A′/B/B′/C
remain unbuilt and are correctly blocked behind that.

**Both blocking design fixes are now DONE and verified (E1d).** `random_move_orders`
and `role_glyphs` live in `src/calibration/` and every organism must use them.
**The prompt is safe to train on.**

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
docs/calibration-program.html      full design spec — hypotheses, controls, ablations, timeline
HANDOFF.md                         this file
src/calibration/
  maze.py       TextMaze, state_bank (paired contrasts), mold_rate manipulation check
  capture.py    affect-free prompt, pooled_resid (last / mean_all / mean_grid), last_token_resid
  analysis.py   3 extraction specs, cosine, split-half, surface_baseline, probe (dual ridge)
  runner.py     set_all_seeds, RunManifest (seed+git SHA+config hash), save_results, summarise
experiments/
  E1a_extraction_spec/     run.py · RESULTS.md · results/*.json · e1a_profiles.svg
  E1b_probe_readout/       run.py · RESULTS.md · results/*.json · e1b_readout.svg
  E1c_functional_gate/     run.py · RESULTS.md · results/*.json · e1c_gate.svg
  E1c2_margin_and_order/   run.py · RESULTS.md · results/*.json
  E1d_corrected_floor/     run.py · RESULTS.md · results/*.json · e1d_signflip.svg
  E1e_margin_averaging/    run.py · RESULTS.md · results/*.json
  E2a_rl_pilot/            RESULTS.md
  E3_reference_extraction/ RESULTS.md · results/*.json
  E4_gate_test/            RESULTS.md · results/*.json · e4_gate_vs_learning.svg
src/calibration/trajectory.py  batched episodes, action-token readout, landing classes
src/calibration/lora.py    minimal LoRA, zero-init B, remove_lora restores exactly
src/calibration/rl.py      single-step Dr.GRPO, group-mean baseline, no std
scripts/
  sync_to_sandbox.sh · plot_e1a.py · plot_e1b.py · plot_e1c.py
```

**Conventions, please keep them:**
- Every experiment is a `run(model, tokenizer, config, out_dir)` returning `(path, results)`.
- Every result JSON carries seed, git SHA, config hash, model, device, versions, timestamps.
- Every experiment gets a `RESULTS.md` with headline / findings / **threats** / next.
- Figures are hand-written SVG — no plotting dependency.
- **Pre-commit expectations in the run.py docstring before running.** This has paid
  off twice already (see §5).

---

## 5. Results so far — both on the untrained model

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

## 8. Known-wrong things to not rediscover

- `\N{...}` unicode escapes for the tile glyphs fail to compile — use literals.
- `accelerate` installed mid-session is invisible to an already-imported
  transformers (it memoises the check). Avoid `device_map`; use `.to("cuda")`.
- transformers 5.14 returns a `BatchEncoding` from `apply_chat_template` — needs
  `return_dict=True` and `**enc`.
- Ridge probes **must** be solved in the dual (`_ridge_dual_predict`). The primal
  form is 2560×2560 at rank ≤134 and did not finish.
- `execute-code.sh --session <id>` goes stale on browser reconnect. Omit it.
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
- **Class-size balancing does not repair a class-composition confound** — in E5 it
  doubled the artefact. Equal sizes are not equal contents.
- 🚩 **`isinstance` is unusable for detecting LoRA wrappers, and this was a live
  bug for most of a day.** Every launcher does
  `del sys.modules["calibration.*"]` then re-imports, which makes `LoRALinear` a
  NEW class object while the wrappers on the resident model are instances of the
  OLD one. `isinstance` → False for all of them, so `has_lora` called a dirty
  model clean and `remove_lora` returned 0 without removing anything. Fixed with
  `is_lora_module` (class name + attribute signature, survives reload). **If you
  add any new LoRA-detecting code, use `is_lora_module`, never `isinstance`.**
  No result was corrupted — `inject_lora` raises when its targets are already
  wrapped, and that loud failure is the only reason the silent half was found.
- **Printing from a kernel-side thread in marimo raises `AssertionError`**
  (`self._stream.cell_id is not None`) — stdout is bound to a cell context a
  detached thread does not have. Any worker thread must wrap its body in
  `contextlib.redirect_stdout(fh)`. E7 lost a full launch to this.
