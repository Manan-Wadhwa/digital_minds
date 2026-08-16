# Fill snippets — each `## KEY` section becomes fill[KEY]

## TABLE2

Table 2. Five pre-registered criteria that passed on the wrong property (all from this program).

| # | run | what the criterion tested | what it should have tested | what happened |
|---|---|---|---|---|
| 1 | E11 dose ladder | that the rate axis had *spread* (sd ≥ 0.05) | that it was *monotone* in reward scale | "RATE AXIS USABLE" printed while ρ = +0.22 (needed ≤ −0.5) |
| 2 | E12 instrument battery | placebo below a fixed 0.5 | placebo below the real instruments (relative bar) | placebo 0.44 passed while every real instrument sat at 0.27–0.50 |
| 3 | E13 organism set | "ORG-C's function may be weaker than ORG-A's" | that stage-two SFT preserved the RL avoidance | a bug (SFT on base-policy moves erased avoidance, 0.23 → 1.05) matched the prediction and would have been reported as interference |
| 4 | E14/E16 v1 narration check | that ORG-B *talks* (aversive rate on adjacent states) | that its talk *tracks the tile* (adjacent − non-adjacent) | 11/12 "valid" ORG-B → 1/12; three seeds emitted the aversive remark on every state |
| 5 | E18 move-mass sweep | "within 0.05 of the control's non-wrecked mean" | a baseline that must exist | the control wrecked 4/4 seeds; the scorer printed `bar n/a` |

## TABLE2_SLIDE

| # | run | tested | should have tested | what happened |
|---|---|---|---|---|
| 1 | E11 | rate axis has spread | rate axis is monotone in dose | "usable" printed while ρ = +0.22 (needed ≤ −0.5) |
| 2 | E12 | placebo < 0.5 | placebo < the real instruments | placebo 0.44 passed; instruments 0.27–0.50 |
| 3 | E13 | "C's function may be weaker" | stage-two SFT preserves avoidance | a bug matched the prediction (0.23 → 1.05) |
| 4 | E16 v1 | ORG-B talks | ORG-B's talk tracks the tile | 11/12 valid → 1/12; 3 seeds aversive on every state |
| 5 | E18 | within 0.05 of control mean | that a baseline exists | control wrecked 4/4; scorer printed `bar n/a` |
