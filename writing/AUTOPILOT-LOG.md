# Autopilot log — paper + slides build (started 2026-08-16)

Operator: Claude (Fable 5, with Opus subagents where noted). User is away; no
questions asked after kickoff. Every file touched, every decision, every run
is listed here in order. Commits are made as work lands (the "don't commit"
instruction was retracted at kickoff: "do whatever, the no commit line is mb").

## Kickoff decisions (user answers)
- Framing: **my call, log the choice** (see §Framing below once decided).
- Compute: **GPU "almost unlimited"** — local is a 4 GB RTX 3050 (useless for
  Qwen3-4B+ work); the molab sandbox at
  `https://sb-fa46f6c1a29f935c.sb.molab.run/` (token given at kickoff, kept out
  of git) has large VRAM. Use it for any new inference/training.
- Git: pull + merge, commits allowed.
- Deliverables: fill the template docx in place (keep styles) →
  `writing/Digital-Minds-Sprint-Submission.docx`; pptx; PDFs of both via
  LibreOffice; appendix as big as needed. `uv`/`pip` installs of python-docx /
  python-pptx OK. Author details were **not** supplied → using
  "Manan Wadhwa" (from git identity) with a placeholder affiliation, flagged
  in the docx for the user to fix.
- Subagents: allowed ("use non-fable models if trusted as subagents … opus").
- "can you check quotas": I have no tool that reads Anthropic usage quotas;
  not checked. Token spend so far is visible only in the session UI.

## 1. Git — pull and merge (done)
- Committed the pre-existing staged docs (reviewer report, compute ledger,
  paper html, template) as `docs: reviewer report, compute ledger, paper html,
  submission template`. Unstaged the two `.pyc` files that had been staged.
- Merged `origin/claude/digital-minds-sprint-strategy-l0b2pz` (one remote-only
  commit `fa991076`, dated 2026-08-13, "propagate E17 and both withdrawals to
  every write-up"). Local branch was 31 ahead and dated 2026-08-14+, i.e.
  strictly newer.
- Conflicts in `README.md` (3 blocks) and `HANDOFF.md` (2 blocks): **kept the
  local (HEAD) side of every block** — the remote side was the older
  2026-08-11/13 status text that local had already superseded (E16 v2, scale
  ladder, VAL01, NAR01c).
- Auto-merged (no conflict) hunks from the remote landed in
  `experiments/E16_calibrated_loading_map/RESULTS.md` (a dated
  "CORRECTION 2026-08-11" block — factual about E16 v1, kept),
  `writing/post-draft.md` (added §9–§10 about the E17 narration diagnosis and
  renumbered later sections — note this draft still carries E16 v1 numbers
  such as I4 −0.971 that E16 v2 later failed to reproduce; the draft is NOT
  the paper source), and the two `docs/*.html` (identical content).
- Merge commit: `4c986490`.

## 2. Reconnaissance (done, ~1 h)
- Three Opus subagents produced scratchpad digests (not in git):
  `factsheet.md` (every quotable number, file:line, ✔JSON re-derived),
  `code_api_notes.md` (adapter round-trip, instruments, loading statistic,
  organism recipes, run harness, JSON schema), `paper_html_digest.md`
  (structure of docs/calibration-program-paper.html + reviewer report).
- Sandbox `sb-fa46f6c1a29f935c` is live: RTX PRO 6000 Blackwell 96 GB, torch
  2.11+cu130, transformers 5.14, **peft missing (not needed — repo has its own
  LoRA)**. `/marimo/repo` still holds the Aug-14 tree incl. **all 60 E16 v2
  adapters**, NAR01a/b/c adapters and SCL01-32B adapters. Qwen3-4B-Instruct-2507
  prefetched into the HF cache (7.6 GB). Token copied to
  `../archive-logs/sb_fa46_token.txt` (outside git, chmod 600).
- Local: `.venv-writing/` (python-docx 1.2, python-pptx 1.0.2, matplotlib) —
  added to .gitignore.

## 3. Framing decision (mine, per kickoff answer "your call, but log it")
**Reframe per the area chair** (docs/reviewer-report.md §"The paper that gets
accepted"), because the data as it stands cannot support the loading-map
paper: the narration axis has 0/12 valid ORG-B (E16 v2), so every "instrument
X does not read narration" sentence is about a 7-organism single-kind (ORG-C)
group. What the data DOES support, and what the paper leads with:
1. **Asymmetry of installability.** A reward-driven avoidance state installs
   reliably (SFT A′ 12/12 at 4B; RL A 6/6 at 14B), but a *narration-only*
   organism could not be installed by any non-degenerate corpus: 5 pool
   sizes × 4 corpus variants × 3 experiments (E17, NAR01a/b/c) + no
   improvement with scale to 32B; only pool=1 (corpus contingency 1.0 by
   construction) clears the bar, 3/8. Co-trained ORG-C reaches contingency
   7/12 and transfers it to a novel glyph (0.70/0.30) where B transfers only
   presence (0.81/0.78). → NEW EXPERIMENT NAR02 (co-training arms) tonight
   to test the one untried lever directly.
2. **Verbal instruments are context-driveable and weight-blind** at every
   size tested (VAL01 5–10 logit swings, bare reads 0.000000; SCL01 verbal
   nulls 1.7B→32B while I1 → d≈1.4 at 14B+). NEW: word-list robustness,
   per-instrument positive controls, difference-of-loadings CIs (reviewer
   blockers 2, 3, 6), the three unreported measures (blocker 8).
3. **Five pre-registered criteria that passed on the wrong property** —
   the transferable methods contribution.
4. Methods/artifact: placebo selection, two-axis screen, organisms +
   adapters, 108-check reproduction script.
The loading map itself is demoted to supporting evidence with the narration
axis explicitly marked as not installed. Title (working): *"The state
installs; the script does not: calibrating AI-welfare instruments against
manufactured organisms."*

## 4. Experiment plan for tonight (all logged below as they run)
- T0 (offline, CPU): d_fn−d_nar seed-clustered CIs (E16 v2 + SCL01 sizes),
  distance-graded narration analysis from `nar_distance` (E16 v2 rows),
  primary scaling named in advance = `residual_measured` (the one every
  RESULTS.md already quotes).
- T1 (sandbox, inference-only on the 60 persisted E16 v2 adapters + 12 D):
  `experiments/v2/PAP01_instrument_robustness` — (a) two extra word lists
  for I2/I4/I6; (b) prompted positive controls (P-NONE/P-AVOID/P-APPROACH)
  on every organism for I2/I4/I5/I7; (c) activation patching C→D, A′→D,
  B→D at the probe layer (+ a small layer sweep) reading I2; (d) residual
  capture at CTX_A/CTX_B for an offline linear-vs-nonlinear cross-organism
  probe.
- T2 (sandbox, training): `experiments/v2/NAR02_cotraining` — ORG-B arms:
  control (E16 recipe), soft-self (policy-preserving co-training:
  soft_move_target=base probs, no anchor), oracle-move (joint CE with a safe
  move, = ORG-C's SFT without RL), 8 seeds, B + B′ per arm + D canary.
  Pre-registered predictions in the run docstring before launch.
- Skipped (logged, not done unless time remains): ENV02 rebuild (blocked on
  C12 + a narration recipe that works in word-world), 14B map re-run.

## 5. Writing pipeline (in progress)
- `writing/paper.md` — the paper source (front matter + sections; `{{KEY}}`
  placeholders for numbers that arrive from tonight's runs).
- `writing/appendix.md` — appendix source; its tables come from
  `scripts/appendix_tables.py` → `writing/appendix_tables.md` (generated
  from committed JSONs, nothing hand-typed).
- `writing/references.md` — 33 references verified by a WebSearch subagent
  (`scratchpad/references.md` has the one-line relevance notes). Notable: the
  "unnamed functional-welfare preprint" the maze is borrowed from is
  Han, Chalmers & Izmailov (2026), arXiv:2605.30232 — verified by its
  pre-training cosine band [−0.23, −0.13] matching the repo's.
- `scripts/make_paper_fill.py` assembles `writing/paper_fill.json` from
  appendix_tables.md + fill_snippets.md + appendix.md + references.md.
- `scripts/build_paper_docx.py` clones the template docx, fills title /
  author / abstract cells in place, deletes the guidance box, and renders the
  markdown body (headings → template Heading 2/3, tables, figures, bullets).
  Author block: replaced the template's 2×3 nested table with one centred
  paragraph (the six narrow cells wrapped a single name badly).
- `scripts/build_slides.py` builds `writing/Digital-Minds-Sprint-Slides.pptx`
  from `writing/slides.md` (13 slides drafted).
- PDFs via `soffice --headless --convert-to pdf`.
- Author affiliation set to "Independent researcher" (placeholder — user did
  not supply one).

## 6. Tonight's experiments — outcomes
- **PAP01** (`experiments/v2/PAP01_instrument_robustness`, 12.1 GPU-min,
  inference-only on the 60 released E16 v2 adapters + 12 ORG-D; sha256
  verified 60/60): word-list verdicts do not survive (4/4 disagree with the
  committed list; no |d| > 0.44; every difference CI spans zero); every
  instrument fires under VAL01's carrier on the trained organisms (true null,
  not floor); patching localises the self-report readout at the final token
  from layer 24 up, identically for B and C; ORG-D reads take exactly 2 values
  over 12 seeds (parity) — effective n for the residual baseline is 1 d.o.f.;
  fp16 adapters reproduce E16 to ~0.1 logit. Residual capture (55 MB) for the
  nonlinear probe was written on the sandbox; local pull was still in
  progress at last check — analysis NOT done, stated as future work.
- **NAR02** (`experiments/v2/NAR02_cotraining`, 39.1 GPU-min): with the
  corpus fixed, no move-token objective installs contingent narration
  (−0.010/+0.038/+0.031/+0.009, 0/8 each); the two "co-train a policy" arms
  collapsed onto a constant move (move entropy caught it; ratio would not
  have). Pre-registration P1 window missed by 0.01 because it was anchored to
  NAR01's 1536-example control — recorded, not adjusted. Follow-up
  **NAR02b** (volume 384/1536 × RL-first yes/no on the ORG-C corpus, 8 seeds)
  requested from the same agent at ~03:00 sandbox time; results pending.
- Offline: `scripts/paper_stats.py` (difference CIs reproduce the committed
  JSON 48/48; Holm; distance-graded narration — geometry note: d≥3 has 9
  states, so the graded contrast is d=1 vs d=2), `scripts/paper_figures.py`
  (F1–F5 + S1–S4).

## 7. Length note
Main text (title → conclusion) renders at ~8 pages in LibreOffice with three
full-width figures and two tables, against the template's recommended 4.
I compressed the prose twice (5.7k → ~3.3k words) and moved everything else
to the appendix; cutting further would drop the pre-registration and
control detail the reviewer report demanded. Flagged for the user to decide.

## 8. Redirect (user, ~03:30): concrete results over softened claims
- User rejected bending the claim; asked where to go. Agreed plan: gold-adjacency
  control (CPU), NAR03 (contrastive remark objective — the attempt to actually
  BUILD the narration-only organism), PAP02 (neutral-glyph transfer with 🟨/🟧
  instead of red/green, yellow narration re-check, path agreement).
- Gold-adjacency control (`scripts/gold_adjacency_control.py`): ORG-C fires on
  penalised-adjacent (0.92) not rewarded-only-adjacent (0.25 vs 0.17 neither);
  per-seed GOLD−NONE −0.02, 11/12 exactly 0; holds on the novel glyph. ORG-B
  flat (0.79/0.77/0.74). Folded into §4.1 and App. J.2 (Table J6b).
- NAR03 (`src/calibration/contrastive.py`, arms in `scripts/nar03_driver.py`,
  reusing NAR02's run.py with a `contrastive` arm key): DPO on chosen/rejected
  remark pairs vs the frozen base + CE on chosen + E16's move anchor. Smoke:
  path wired, anchor on; margin negative after 12 steps (CE lift dominates
  early) — gate relaxed to "path ran"; added a weight-5 arm. Running 4 shards
  (seeds 0-1/2-3/4-5/6-7), arms control/dpo(β.1)/dpo_hi(β.5)/dpo_w5(β.1,w5),
  B+B′ per arm + D canary. Monitor armed.
- PAP02 one-seed probe launched; full run after NAR03 to avoid contention.
- Killed all leftover polling shells and the residual puller (user request);
  the two subagents were stopped by the user; monitors only from here on.
- NAR02b (volume × RL-first) shards may still be running on the sandbox from
  the stopped agent; a Monitor watches its status files.

## 9. Power loss, sandbox lost, final build (2026-08-16)

- Operator power failure while NAR03 (4 shards, 8 seeds, 4 arms) and NAR02b
  (2 shards) were running. On resume (04:16 UTC) the sandbox still answered:
  status files present, 9 driver processes alive, no DONE/ERROR markers. On
  the next check the sandbox returned HTTP 410 (gone) — the molab box expired
  and every unpulled file with it. Nothing from NAR03/NAR02b was recovered.
- Decision (no bending of claims): the paper reports NAR03 and NAR02b as
  pre-registered + launched + lost, with the design and the smoke-run
  observation only (Appendix J.3 addendum, Appendix N, Future work). No number
  from either is claimed. Code, tests and drivers are in the repo; they are
  the first thing to rerun on a fresh sandbox.
- Fixed a stale sentence in Table J6b notes ("yellow re-measurement queued" →
  points to PAP02 result). Removed empty nested NAR02B_* placeholders (they
  survived to the PDF as literal `{{...}}`).
- Rebuilt: paper_fill.json → docx (26 pp incl. appendix) → PDF; pptx (13
  slides) → PDF. Zero unfilled placeholders in either PDF.
- Not done for lack of compute: NAR03, NAR02b, PAP01 residual probe transfer,
  14B map. All are listed under Future work.

## 10. Sandbox volume recovered; NAR02b + NAR03 results; NAR03b launched (2026-08-16 15:20 UTC)

- New sandbox `sb-05e7b0cf547c7d50` mounts the SAME /marimo volume: NAR03 (4
  shards) and NAR02b (2 shards) had finished at 04:21–04:23 / 03:41 before the
  old sandbox went 410. Pulled the JSONs (adapters left on the volume) into
  `experiments/v2/NAR02_cotraining/results_nar03/` and `results_b/`.
- NAR02b (score_nar02.py): Q1 FAIL (sft1536 2/8), Q2 FAIL (rl_sft384 2/8),
  Q3 PASS (rl_sft1536 6/8); contingency tracks avoidance across 32 organisms
  (r 0.69); no organism contingent AND policy-invariant. Folded into §4.1
  (NAR02_RESULTS), abstract/contrib sentences, App. J.3 Table J7b.
- NAR03: contingency 0.00 every arm, 0/8; margin positive but greedy remark
  unmoved; presence rose to 0.96–0.99 both classes. App. J.3 Table J7c.
  Retracted the "launched-and-lost" wording from §9 in the paper text.
- NAR03b escalation launched: arms control3 / dpo_b1_w5 / dpo_b2_w20 /
  dpo_b1_w10_ce02, 3 epochs, 8 seeds; shards b1a,b1b on sandbox 05e7 and
  b1c,b1d on the second sandbox `sb-9a14f47831f84130` (user-provided;
  repo synced, model will download). Driver gained NAR03_ARMS/NAR03_EPOCHS
  env overrides. Monitor armed; paper carries `{{NAR03B_*}}` placeholders
  with a "running at submission" default.
- Build scripts: placeholder substitution now 4-deep (docx) / 3-deep (slides)
  so addenda inside appendix sections resolve.

## 11. NAR03b/c, NAR04/NAR04b; second sandbox loss; paper closed (2026-08-16/17)

- NAR03b (β 1–2, w 5–20, CE 1/0.2, 3 epochs) and NAR03c (CE 0/0.05): every
  arm that moved the margin deleted the remark (presence 0/0, outputs = bare
  move word); contingency 0.00, 0/8 all arms. Paper: App. J.3 Table J7d,
  NAR02_RESULTS paragraph, abstract/contrib sentence.
- NAR04 (new `src/calibration/remark_rl.py`): GRPO-style RL on the remark,
  class-match reward, move anchor kept, warm start from control SFT. Pilot
  (seeds 0,1,4,5): groups class-homogeneous → zero advantage → contingency 0.
  NAR04b adds one injected exemplar per class per group (on-policy
  unlikelihood on the model's own wrong-class remarks): reward stays at the
  0.635 marginal; mean contingency ≤ +0.06; best seeds +0.23/+0.22; 0/8.
  Paper: Table J7e, §4.1 paragraph, slide note. Two OOMs (5 drivers/GPU) fixed
  by chunked log-softmax; an `on_pol` ordering bug fixed after one failed
  launch.
- Sandbox 05e7 went 410 (second loss) taking NAR04b seeds 0–3 and pilot
  seeds 0,1; seeds 0–3 rerun on 9a14 (e1a/e1b) — the paper reports 4-seed
  NAR04b numbers marked "seeds 4–7; 0–3 rerunning" and will be updated to 8
  seeds if they land before submission.
- Bug found in the Monitor filter (grep'd only the last status line, missed
  DONE lines followed by SUMMARY) — fixed by grepping the whole file.
- Compute lines updated (App. N). Docx substitution now 6-deep.
- Final build: docx 28 pp incl. appendix, PDF; slides 13, PDF; zero unfilled
  placeholders in either.
