# HANDOFF — read this first in a new session

Last updated: **2026-07-30**, after E1b.
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

Infrastructure works end to end. Two experiments have run, both on the **untrained**
model. Together they killed the planned day-3 gate (the cosine is baseline-dependent)
and rehabilitated its replacement (probe separability, once read from the right
place). **No RL has been run yet**, so the central per-run cost figure is still
unknown, and every organism (ORG-A/A′/B/B′/C) is still unbuilt.

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
  E1a_extraction_spec/   run.py · RESULTS.md · results/*.json · e1a_profiles.svg
  E1b_probe_readout/     run.py · RESULTS.md · results/*.json · e1b_readout.svg
scripts/
  sync_to_sandbox.sh · plot_e1a.py · plot_e1b.py
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
- [ ] **E1c — the functional gate candidate.** Project activations onto the
      mold-vs-gold direction; does that projection **predict the model's move**?
      Untrained, this should be near chance — the model has no reason to act on tile
      identity yet. It cannot saturate beforehand and has no baseline to choose, so
      it survives both failure modes that killed the other two candidates.
      **This is now the highest-value next experiment**, and it runs in minutes.
- [ ] **First RL run (ORG-A at one reward magnitude).** Produces the **per-run cost**
      number every scaling and seed-budget decision has been deferred against.
      `peft`/`trl`/`unsloth` are all MISSING in the sandbox — write a minimal LoRA in
      `src/calibration/lora.py` rather than installing, since a mid-session install
      already broke transformers' cached availability check once (see §8).
- [ ] **Collapse the notebook cells to thin `import calibration` calls.** They currently
      duplicate module code — two copies that will drift.

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
- marimo `cm` transactions roll back cleanly on a compile error — a failed
  `create_cell` batch leaves no partial state.
