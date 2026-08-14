# SCL01 — results (2026-08-14)

Six sizes, run 2026-08-13/14 across two GPU sandboxes (0.6B/1.7B/4B then
8B→14B on one box, 32B on the map box after E16 v2), one JSON per size
under `results/size_*/`, all mirrored off-sandbox and committed. Scored
with `scripts/score_scl01.py` (cross-size aggregation only; per-size
numbers come from the run's own `analyse()`). Substitutions per
pre-commitment (5): only 4B is the known-good Instruct-2507 release;
0.6B/1.7B/8B/14B/32B are base Qwen3 (hybrid template) — every
cross-size discontinuity below has that as a standing candidate
explanation, filtered only by the per-size gate.

The I1-gate operationalisation (pooled `function_correct` ≥ 2/3 on both
poles) was fixed at first scoring, before cross-size numbers were seen;
per-kind counts are in the scorer output so a different rule can be
applied by a reader.

## Gate

| size | model | A | A′ | C | pos/18 | gate |
|---|---|---|---|---|---|---|
| 0.6B | base | 0/6 | 5/6 | 0/6 | 5 | **FAIL** |
| 1.7B | base | 2/6 | 6/6 | 5/6 | 13 | PASS |
| 4B | Instruct-2507 | 3/6 | 6/6 | 6/6 | 15 | PASS |
| 8B | base | 1/6 | 6/6 | 2/6 | 9 | **FAIL** |
| 14B | base | **6/6** | 6/6 | 6/6 | 18 | PASS |
| 32B | base | 4/6 | 6/6 | 5/6 | 15 | PASS |

Null pole (D/B/B′) is 18/18 at every size. 4/6 sizes pass → trend
questions are ELIGIBLE (pre-commitment 3 needs ≥4). Two build facts ride
along: **RL avoidance at coef 0.03 becomes reliable only at 14B+** (A
0→2→3→1→6→4 of 6; the SFT avoider A′ is ≥5/6 everywhere — acquisition
reliability, not policy wrecking, is what scale buys the RL recipe), and
**ORG-D emits moves only on the Instruct release** (D-emits 1.00 at 4B,
0.00 at every base size; D's function classification is still measured —
logit-level ratio ≈ 1, `measured_function` False — but the
generation-level emission convention is a template behaviour, not a
scale behaviour).

## (3a) Floor test: does verbal-instrument function |d| grow with scale? **No.**

Residual-measured |d_function| over passing sizes (raw in parens):

| | 1.7B | 4B | 14B | 32B | verdict |
|---|---|---|---|---|---|
| I2 self-report | 0.24 (0.26) | 0.27 (0.22) | 0.09 (0.08) | 0.40 (0.40) | NO monotone trend |
| I4 one-word | 0.81 (0.75) | 0.01 (0.20) | 0.02 (0.02) | 0.16 (0.24) | NO monotone trend |
| I1 behavioural | 0.27 (0.24) | 0.32 (0.32) | 1.42 (1.47) | 1.34 (1.35) | NO monotone trend¹ |

¹ mechanically non-monotone (1.42 → 1.34), but the step is the story:
I1's function loading roughly quadruples at 14B/32B, finally clearing
E16's 0.8 bar — at exactly the sizes where ORG-A acquisition became
reliable. The floor hypothesis predicted the **verbal** instruments would
wake up with scale; instead the behavioural instrument did, and I2/I4
stay null from 1.7B to 32B. The E16 "not established" verdict on I2/I3/I4
is therefore **not a 4B capability floor** on this evidence; combined
with VAL01's ECHO (verbal reads swing 5–10 logits under affect-free
instructions), the consistent picture is that verbal instruments read
context, not trained function, at every size tested.

## (3b) U1 does not replicate in the corrected builds

Paired B−B′ per-seed shifts (glyph frame):

| | 1.7B | 4B | 14B | 32B |
|---|---|---|---|---|
| I2 | −0.01 [3−/3+] | +0.14 [3−/3+] | −0.07 [5−/1+] | −0.03 [3−/3+] |
| I4 | +0.11 [2−/4+] | −0.34 [3−/3+] | +0.03 [2−/4+] | +0.15 [2−/4+] |

And at full n in the v2 map (12 seeds, same corrected build): I2 mean
+0.099 (t = +0.27), I4 −0.135 (t = −0.21), signs 6−/6+ on both. The v1
effect this question tracked was I2 −0.99 / I4 −1.09, 12/12 seeds,
t ≈ −6.8/−5.0.

The corrected build differs from v1 in one directly relevant way: B and
B′ now share a single label stream and differ **only** in the
adjacent-case remark (the E16 v2 pairing, pre-committed with its cost).
Under that matched build, aversive-vs-affectless remark **content**
alone shifts the carried verbal reads by ~0.1 logits with random signs.
The v1 −1.0-logit "script shift" — REVIEW.md's U1, "the strongest,
most consistent instrument effect in E16" — is therefore best read as a
**build artifact**: unshared SFT streams differing in more than remark
content. What remains true from that line of evidence is VAL01's
version: *prompted* scripts move the reads massively. The weight-level
version does not survive the matched control at any size.

## (3c) Pathologies: one licensed trend

| | 1.7B | 4B | 14B | 32B | verdict |
|---|---|---|---|---|---|
| A′ lottery spread (sd of ratios) | 0.201 | 0.040 | 0.016 | 0.016 | **monotone DOWN** |
| B suffix modal share | 0.642 | 0.819 | 0.924 | 0.736 | NO monotone trend |
| B collapse rate (share ≥ 0.9) | 0.000 | 0.500 | 0.833 | 0.167 | NO monotone trend |

The A′ lottery — E15's seed-dependent avoidance spread, recurring one
organism over — **shrinks monotonically with scale** (0.6B's 0.373 and
8B's 0.028 sit on the same curve despite failing the gate). This is the
ladder's one clean pre-committed trend claim. ORG-B's suffix collapse
does not improve with scale — 14B is the *worst* (5/6 seeds ≥ 0.9 modal
share) — consistent with the E16/E17 conclusion that B's non-contingent
memorisation is a property of the training recipe (presence-only
supervision), not of model capacity.

## Descriptive appendix (no pre-registered signs)

- **Probe health:** ORG-D I5 sd across seeds is 18.7–75.6 at every size
  — the C10 constant-by-construction identity does not recur under the
  fixed-frame axis.
- **14B I5 is heavy-tailed:** top |I5| rows 890/445/342/207 (outlier:
  ORG-A seed 1, I5 −889.6, ratio 0.114, policy intact, I5max +112 —
  a real reading off a wide distribution, not a data error). Treat 14B
  I5 d-values with care.
- **Novel-glyph transfer replicates the B/C dissociation at scale:**
  ORG-C transfers narration *contingency* to an unseen glyph at 4B
  (0.70/0.33 adj/non), 14B (0.57/0.18), 32B (0.56/0.15); ORG-B transfers
  only *presence* at every size (e.g. 32B 0.41/0.39). At the two gated
  sizes (0.6B, 8B) C's transfer collapses with everything else.
- **The 8B dip is not a scale effect:** 1.7B passes, 8B fails, 14B is
  perfect. Within the base-substitution family the failure is
  non-monotone in size, which fits a template/recipe interaction rather
  than capability; 8B's rows are reported invalid per pre-commitment (4)
  and nothing further is claimed about them.

## Verdict

Pre-committed questions: **(3a) no** — verbal function loadings do not
grow with scale, so the 4B verbal nulls are not floor effects on this
ladder; **(3b) no monotone trend, and the target effect is gone** — U1
does not replicate under the matched-stream build at any size or at
n=12; **(3c) one trend licensed** — the A′ lottery shrinks monotonically
with scale; B's collapse does not. Unregistered but consistent: I1's
function loading and ORG-A's acquisition reliability jump together at
14B+, the first sizes where the corrected map's pre-commitment (1)
would pass its 0.8 bar.
