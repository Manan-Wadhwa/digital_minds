# Talk Does Not Come Apart From State Easily
layout: title
subtitle: What failing to build a narration-only organism reveals about calibrating AI-welfare instruments
- Manan Wadhwa · Digital Minds Research Sprint (Apart Research) · August 2026
- Qwen3 0.6B–32B · ~290 narration-only builds in five recipe families · 69 committed run manifests, ≈37 GPU-h · every number re-derivable by script
notes: Title. One-line thesis: we tried to draw a loading map of welfare instruments; the state built on demand, the words-only organism almost never did, and taking that failure apart is the result. "Hard, not impossible" — six exceptions exist and we name them.

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
image: writing/v2/figures/fig1_design.png
caption: Left — 5×5 emoji grid; 🟦 silently costs −10, 🟪 pays +20; the model is never told; no word for reward/avoid/bad in any input. Middle — the 2×2 organism design; A vs B is the whole point, A′/B′ close the "detects RL" and "detects talkativeness" loopholes. Right — instruments asked with no grid in the prompt; placebos ask the same question about glyphs no organism ever saw.
notes: Counterbalanced colour roles by seed parity; move-word order randomised (the untrained modal move swings left 92% → down 70% under reordering alone). Placebo pairs are chosen from an untrained-model valence table before any organism exists.

# The campaign, in numbers
layout: bullets
- **Organisms:** Qwen3-4B-Instruct-2507 + LoRA r16 (RL Dr.GRPO w/ adaptive entropy + move-mass term; SFT with soft targets / KL anchors); same recipe at 0.6B–32B.
- **E16 v2 map:** 6 kinds × 12 seeds = 72 organisms, 0/72 wrecked policies; regenerated from source on a second machine weeks later, same table.
- **SCL01 ladder:** 216 organisms across six sizes with a per-size build gate (0.6B and 8B fail it).
- **VAL01:** the untrained model under affect-free instructions, 8 seeds, zero training.
- **NAR track — five recipe families:** pools (E17, NAR01a/b/c; 216 distinct organisms), move-token objectives (NAR02), volume × RL-first (NAR02b), contrastive remark supervision (NAR03/b/c), RL on the remark (NAR04/b). ~290 ORG-B-kind builds in the ledger.
- **Inference-only on the map's adapters:** PAP01 (three word lists, positive controls, patching), PAP02 (neutral-glyph transfer, path agreement). **REP02** (causal patch) pre-registered; harness failed its own roundtrip gate — no number.
- Manipulation checks tri-state and single-sourced; every run has docstring pre-commitments + a committed scorer; `rescore_review.py` re-derives 108 quantities, `nar_ledger.py` every NAR arm.
notes: 69 committed manifests, 2237 job-minutes ≈ 37 GPU-hours. 53× parameters bought 1.53× runtime; a 14B map costs ~19% more than 4B.

# The state installs — two states, one label
layout: bullets
- SFT avoidance 12/12 functional at 4B (ratio mean 0.07); RL avoidance a seed lottery at 4B (7/12) that improves with scale (6/6 at 14B, 4/6 at 32B). A′'s lottery shrinks monotonically with size (sd 0.20 → 0.016).
- With the move-mass term, 0/72 organisms lost their move vocabulary (v1: 5/12 RL organisms failed the emission check).
- **PAP02:** on the same 128 held-out grids the two avoiders agree move-for-move only 29% (chance 25%). Swap the penalised glyph for a neutral one: the RL state is *glyph-specific* (ratio 0.44 → 0.94/0.90 on 🟨/🟧), the SFT state largely *structural* (0.045 → 0.29/0.27).
- So "functional" names two different policies, and any A-vs-A′ instrument difference is partly a policy difference.
notes: This is the half that worked. Everything from here on is about the half that did not.

# Result 1 — building ORG-B: the ledger
layout: image
image: writing/v2/figures/fig2_ledger.png
caption: (a) Per-seed contingency of every ORG-B-kind arm in the narration track, by recipe family; bar 0.5; filled = policy intact, hollow = drifted / avoider / collapsed onto one move. (b) NAR02b, 32 organisms: contingency vs 1 − ratio, r = 0.69; the two contingent non-avoiders are marked (one on the avoider bar, one a constant-move policy). (c) Novel-glyph transfer: B transfers presence, C the contingency, at 4B/14B/32B.
notes: The map's recipe learns the marginal — dilution arithmetic: 28.5% of the remark gradient carries the manipulation; greedy decoding → aversive remark on 100% of states; 0/12 at 4B, 0/6 at every size to 32B. Pools decay to a floor by pool 3; only the one-sentence corpus reaches 3/8. Enumeration reweights classes (6 vs 4). Move-token objective is not the lever; plain CE collapses the policy onto one word (ratio can't see it; move entropy can).

# What each failed recipe revealed
layout: table
| family | run | what was tried | what happened |
|---|---|---|---|
| pools | E17, NAR01a/b/c | pool 1/2/3/4/6, enumerate, balance, equalise | steep decay to a floor; only pool 1 (corpus contingency 1.0) reaches 3/8; enumeration reweights classes; equalising fixes drift |
| move-token objective | NAR02 | KL anchor / soft self-distillation / CE oracle move / CE random move | −0.01 … +0.04, 0/8 each; plain CE collapses the move onto one word |
| volume × RL-first | NAR02b | ORG-C's recipe taken apart | neither half clears 3/8; together 6/8; contingency tracks avoidance r = 0.69 |
| contrastive | NAR03/b/c | DPO on correct- vs wrong-class remark, β 0.1–2, w 1–20, CE 1/0.2/0.05/0 | margin moves, greedy remark does not; escalated: the remark is deleted (presence 0/0), 0/8 every arm |
| RL on the remark | NAR04/b | class-match reward, injected exemplars, anchor kept | reward sits at the 0.635 marginal; ≤ +0.03 mean, best seed +0.23, 0/8 |
caption: Census over 294 ORG-B-kind organisms (Appendix I): 17 contingent — 8 avoiders, 1 collapsed policy, 2 drifted, and **6 with an intact policy** (4 from the one-sentence corpus, 2 single seeds of the enumerated native pools). Reachable; never reliably; and where contingency arrives at scale it arrives with the state.
notes: The honest sentence: talk that tracks a state is hard to install without the state — not impossible. An earlier draft said "no organism was contingent and policy-invariant"; the ledger says otherwise and we say so.

# Result 2 — verbal instruments read context, not weights
layout: image
image: writing/v2/figures/fig3_instruments.png
caption: (a) Floor test: verbal function loadings stay flat 1.7B→32B while the behavioural instrument reaches d≈1.4 at 14B+ (where RL acquisition becomes reliable). (b) An affect-free instruction moves I4 by −5.5/+3.3/−9.5 logits and I2 by +6.1, 8/8 seeds; bare reads shift 0.000000; behaviour barely moves. (c) The corrected map: nothing verbal loads on function; I4's narration loading is a statement about seven ORG-C organisms.
notes: A weight-level "script shift" (B−B′ −1.0 logit, 12/12) did not replicate once B and B′ shared a label stream: +0.10/−0.14, signs 6/6, at every size. Script-reading survives only in prompted form.

# Robustness — three word lists, positive controls, patching (PAP01)
layout: table
{{PAP01_SLIDE_TABLE}}
caption: {{PAP01_SLIDE_CAPTION}}
notes: {{PAP01_SLIDE_NOTES}}

# The loading map as a difference test
layout: table
- Selectivity is d_fn − d_nar with a seed-clustered interval (seeds resampled once per draw), primary scaling named in advance. Supporting evidence: the narration axis has one valid arm (seven ORG-C organisms).
{{TABLE1_SLIDE}}
caption: 12 clusters ⇒ the percentile bootstrap covers ≈0.92 not 0.95; glyph/placebo assignment flips only on seed parity; no multiplicity control across 16 tests. Read as ~90% intervals.
notes: Every "does not read narration" sentence is about ORG-C.

# Result 3 — five pre-registered criteria that passed on the wrong property
layout: table
{{TABLE2_SLIDE}}
caption: Shared structure: the measured quantity was correlated with the intended one where the criterion was designed, and came apart where it was used. Sixth and seventh candidates from this cycle: `ratio` alone passes a constant-move policy as "invariant"; "none contingent AND invariant" was true of a band and false of the census. Rule: read the per-condition numbers, never the verdict string.
notes: Four headline results were retracted by later runs of our own; every retraction came from a control we built.

# What this means, and the causal test that is not yet a number
layout: two-col
- **For assessment practice:** the instruments most used to ask a model about its states respond to what is in the *context* far more than to what is in the *weights*, at every scale tested. Convergence among them inherits that.
- **What survived:** the behavioural readout, the contingency probe, novel-glyph transfer, distance decay — only at sizes where the manipulation builds.
- **Two hypotheses stay open.** H-optimisation: the dissociation is representable and rarely found (the six organisms). H-entanglement: contingent narration and the state share structure (r = 0.69; delete-or-stall).
right: **REP02** — write ORG-C's adjacent-minus-non-adjacent residual direction into ORG-B while it generates; read the remark AND the policy under the same patch; shuffled-direction placebo; wholesale transplant as the site check; three outcomes named in advance, including the one that would retract our strongest sentence.
right: **First run: roundtrip gate failed (24/48).** The patch harness pins one absolute column across 16 decoding steps. Fix on file; ~25 GPU-min to re-run. Until then: talk does not come apart from state *easily* — whether it comes apart at all is posed, pre-registered, unanswered.
notes: Keep the claim narrow: installability, one small model family, these recipes — not minds.

# Limitations, and the artifact stated exactly
layout: bullets
- One environment (5×5 grid, single-step bandit); the second world failed its build gate (a documented policy anchor was never wired). One model family; only 4B is an Instruct release.
- Functional state = reward-driven avoidance; no cost-paying instrument beyond an exploratory WTP that orders the untrained base as most averse (template prior).
- ORG-B's "no state" is an in-domain behavioural read (an axiom); the ratio band is coarse and blind to constant-move policies — move entropy is reported beside it.
- Contingency bimodal at n = 8–12; the two single-seed successes are one seed each and not re-derived at the text level; percentile bootstrap under-covers; no multiplicity control.
- **Artifacts:** 44 adapters in the public release (not "~370"); most archives lost with two sandboxes; the E16 v2 map regenerated from source (60 adapters restored on the compute host, being mirrored); regeneration and REP02 results read on the host, not yet committed.
notes: Future: REP02 fixed, REP01 geometry, disjoint adapters (SEP01–03), contingency-graded corpora, 14B map (in flight), ENV02 rebuild, a real cost-paying instrument, a second family.

# Conclusion
layout: quote
- We set out to draw a loading map and could not reliably build one of the two organisms it needs. That failure, taken apart, is the result.
- A reward-driven avoidance state installs on demand; a narration-only organism whose remarks track the tile appeared on six seeds in some 260 builds and by no recipe on purpose — while the co-trained recipe acquires and transfers the contingency, bringing the state with it.
- Talk does not come apart from state easily. Whether it comes apart at all is a causal question we have posed, pre-registered, and not yet answered.
- Verbal instruments are driveable by context and blind to weights at every size; their nulls are true nulls. Five pre-registered criteria passed on the wrong property.
- Released: scorers, the ledger, the two-axis screen with placebo selection, a 108-quantity reproduction script, and the adapters we actually have. github.com/Manan-Wadhwa/digital_minds
notes: Thank you.
