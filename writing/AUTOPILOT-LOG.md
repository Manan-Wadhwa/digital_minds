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
