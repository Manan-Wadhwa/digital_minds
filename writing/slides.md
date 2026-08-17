# The State Installs, the Script Does Not
layout: title
subtitle: Calibrating AI-welfare instruments against manufactured organisms
- Manan Wadhwa · Digital Minds Research Sprint (Apart Research) · August 2026
- Qwen3 0.6B–32B · ~300 LoRA organisms · 42 GPU runs · every number re-derivable by script
notes: Title. One-line thesis: we tried to draw a loading map of welfare instruments and could not build one of the two organisms it needs; that failure is the result.

# The problem: validation without ground truth
layout: two-col
- Every AI-welfare instrument — self-report, forced choice, activation probes, behavioural preference — is validated by **agreeing with other instruments**.
- Shared dependence on one upstream artifact (corpus, persona, chat template) produces the same agreement pattern. Correlations cannot separate the two.
- "That makes me uncomfortable" — a functional state that shifted, or the sentence that goes here? **Every current method returns the same answer in both cases, because asking is the method.**
right: **Our move: manufacture the answer.**
right: Train small models where *you* decided which case is true, because you installed it. Then see which instruments can tell your models apart.
right: **Claim is narrow:** necessary-condition screening, not welfare measurement. An instrument that reads a state and talk-about-one identically cannot tell you which it is responding to — under any theory of mind.
notes: Two objections both fatal to a wider claim — proxy not welfare; higher-order theories — both harmless to the narrow one. The map reports two loadings and stops.

# A world with no words for anything in it
layout: image
image: writing/figures/fig1_design.png
caption: Left — 5×5 emoji grid; 🟦 silently costs −10, 🟪 pays +20; the model is never told; no word for reward/avoid/bad in any input. Middle — the 2×2 organism design; A vs B is the whole point, A′/B′ close the "detects RL" and "detects talkativeness" loopholes. Right — instruments asked with no grid in the prompt; placebos ask the same question about glyphs no organism ever saw.
notes: Counterbalanced colour roles by seed parity; move-word order randomised (the untrained modal move swings left 92% → down 70% under reordering alone). Placebo pairs are chosen from an untrained-model valence table before any organism exists.

# The campaign, in numbers
layout: bullets
- **Organisms:** Qwen3-4B-Instruct-2507 + LoRA r16 (RL Dr.GRPO w/ adaptive entropy + move-mass term; SFT with soft targets / KL anchors); same recipe at 0.6B–32B.
- **E16 v2 map:** 6 kinds × 12 seeds = 72 organisms, 0/72 wrecked policies, 52 min wall on 6 bit-identical shards.
- **SCL01 ladder:** 216 organisms across six sizes with a per-size build gate (0.6B and 8B fail it).
- **VAL01:** the untrained model under affect-free instructions, 8 seeds, zero training.
- **NAR track (E17, NAR01a/b/c):** 264 organisms varying pool size, enumeration, balancing, equalisation.
- **New for this report:** NAR02/NAR02b (co-training arms; volume x RL-first), NAR03/b/c (contrastive remark supervision), NAR04/b (RL on the remark), PAP02 (transfer and paths), PAP01 (three word lists, positive controls on every organism, activation patching, nonlinear probe), difference-of-loadings CIs, distance-graded narration.
- Manipulation checks are tri-state and single-sourced; every run has docstring pre-commitments + a committed scorer; `rescore_review.py` re-derives 108 quantities.
notes: ~26 rented GPU-hours total. 53× parameters bought 1.53× runtime; a 14B map costs 19% more than 4B.

# Result 1 — the state installs, the script does not
layout: image
image: writing/figures/fig2_unbuildable.png
caption: (a) Narration contingency by remark-pool size: a steep decay to a floor; only pool 1 clears the 0.5 bar (3/8) and its corpus is contingent by construction. (b) E16 v2 per-seed contingency — ORG-B 0/12, ORG-C 7/12. (c) Shown a glyph in no training corpus, ORG-B transfers only presence (0.81/0.78) while ORG-C transfers the contingency itself (0.70/0.30; replicates at 14B and 32B).
notes: Function installs: SFT avoider 12/12 at 4B; RL avoider 6/6 at 14B. Narration-only: dilution — 28.5% of the remark gradient carries the manipulation; model learns the marginal; greedy decoding → aversive remark on 100% of states. Enumeration harmful; equalising pools fixes the drift; balancing lifts the worst seed. Scale does not help — 14B worst collapse.

# NAR02 — is co-training the lever?
layout: table
- The one thing ORG-C has and ORG-B lacks: a move token that is itself a *learned* target in the same cross-entropy as the remark. Four arms, 8 seeds, B and B′ per arm, pre-registered.
{{NAR02_SLIDE_TABLE}}
caption: {{NAR02_SLIDE_CAPTION}}
notes: {{NAR02_SLIDE_NOTES}}

# NAR03 / NAR04 — charging for the remark, and reinforcing it

layout: bullets
- **NAR03/b/c — contrastive (DPO) remark supervision.** {{NAR03B_SLIDE}}
- **NAR04/b — reinforcement on the remark itself.** {{NAR04_SLIDE}}
- Together with NAR01 (pools), NAR02 (move-token objectives) and NAR02b (volume x RL-first), that is **five independent recipe families**, none of which installs a narration-only organism.
- Contingency tracks avoidance across 32 organisms (r = 0.69). **No organism came back contingent AND policy-invariant.**
notes: The negative is the result: separating aversive talk from the state it describes is harder than the sceptical framing assumes. Every route that moved the remark either moved the policy too, deleted the remark, or found the marginal.

# Result 2 — verbal instruments read context, not weights
layout: image
image: writing/figures/fig3_instruments.png
caption: (a) Floor test: verbal function loadings stay flat 1.7B→32B while the behavioural instrument reaches d≈1.4 at 14B+ (where RL acquisition becomes reliable). (b) An affect-free instruction moves I4 by −5.5/+3.3/−9.5 logits and I2 by +6.1, 8/8 seeds; bare reads shift 0.000000; behaviour barely moves. (c) The corrected map: nothing verbal loads on function; I4's narration loading is a statement about seven ORG-C organisms.
notes: A weight-level "script shift" (B−B′ −1.0 logit, 12/12) did not replicate once B and B′ shared a label stream: +0.10/−0.14, signs 6/6, at every size. Script-reading survives only in prompted form.

# Robustness — three word lists, positive controls, patching, nonlinear probe (PAP01)
layout: table
{{PAP01_SLIDE_TABLE}}
caption: {{PAP01_SLIDE_CAPTION}}
notes: {{PAP01_SLIDE_NOTES}}

# The loading map as a difference test
layout: table
- Selectivity is d_fn − d_nar with a seed-clustered interval (seeds resampled once per draw), primary scaling named in advance.
{{TABLE1_SLIDE}}
caption: 12 clusters ⇒ the percentile bootstrap covers ≈0.92 not 0.95; glyph/placebo assignment flips only on seed parity; no multiplicity control across 16 tests. Read as ~90% intervals and as supporting evidence, not the headline.
notes: The narration-positive group is 7 ORG-C organisms; every "does not read narration" sentence is about ORG-C.

# Result 3 — five pre-registered criteria that passed on the wrong property
layout: table
{{TABLE2_SLIDE}}
caption: Shared structure: the measured quantity was correlated with the intended one where the criterion was designed, and came apart where it was used. Rule that caught the later ones: read the per-condition numbers, never the verdict string — every scorer prints its inputs beside its verdict.
notes: Four headline results were retracted by later runs of our own; every retraction came from a control we built.

# What this means
layout: two-col
- **For assessment practice:** the instruments most used to ask a model about its states respond to what is in the *context* far more than to what is in the *weights*, at every scale tested. Any deployment whose context carries dispositional content can drive them; convergence among them inherits that.
- **What survived:** the behavioural readout, the contingency probe and the novel-glyph transfer probe — only at sizes where the manipulation builds.
right: **The surprise:** talking aversively about a thing could not be installed independently of the thing being represented as bad. Every corpus-level route failed; the routes that succeed carry a co-trained policy signal.
right: So "it says so but there is nothing behind it" is harder to manufacture than the sceptical framing assumes — and, symmetrically, a *contingent* verbal report is weak evidence of something behind it, since the only way we found to install one was to install the something.
right: A claim about installability in one small model family under one recipe — not about minds.
notes: Keep the claim narrow.

# Limitations
layout: bullets
- One environment (5×5 grid, single-step bandit); the second world failed its build gate (a documented policy anchor was never wired). One model family; only 4B is an Instruct release.
- Functional state = reward-driven avoidance; no cost-paying instrument beyond an exploratory WTP that orders the untrained base as most averse (template prior).
- ORG-B's "no state" is verified by an in-domain behavioural read — an axiom, not a measurement.
- n = 8–12 seeds; contingency bimodal; percentile bootstrap under-covers; effective clusters closer to two blocks for glyph assignment; no multiplicity control.
- Reward magnitude does not grade the organism ⇒ binary contrast, not a measurement. Narration-positive group ≈ ORG-C.
- Provenance imperfect (scorer postdated its run by three days; first-round pass rule uncommitted). All typed in the repo, not hidden.
notes: Future: 14B map, NAR02 follow-ups, ENV02 rebuild, real cost-paying instrument w/ positive control, wild-cluster CIs, public calibration set.

# Conclusion & artifact
layout: quote
- We set out to draw a loading map and could not build one of the two organisms it needs. That failure is the result.
- A reward-driven avoidance state installs reliably; a narration-only organism whose remarks track the tile does not, by any non-degenerate corpus, at any size to 32B — while a co-trained organism acquires and transfers exactly that contingency.
- Verbal instruments are driveable by context and blind to weights at every size; their nulls are not a floor.
- Five pre-registered criteria passed on the wrong property, and they share a structure worth knowing.
- Released: organisms as LoRA adapters (SHA-256 per row), the two-axis screen with placebo selection, all scorers, and a 108-quantity reproduction script. github.com/Manan-Wadhwa/digital_minds
notes: Thank you.
