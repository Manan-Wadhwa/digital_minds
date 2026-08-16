# paper_stats — every number recomputed from the committed JSONs

E16 v2 map: `experiments/E16_calibrated_loading_map/results/20260814T021231Z_95e6e2b8e2df.json`  (72 organisms, 60 trained, 12 seeds)

Bootstrap: seed-clustered, 2000 draws, torch-MT19937 stream reproduced in numpy, percentile `[int(.025n), int(.975n)-1]`; `check` compares the recomputed `d` and `ci_clustered` against the committed JSON.

## 1. Selectivity: d_function − d_narration (E16 v2)

Primary scaling `residual_measured` (named as primary in advance: `score_e16.py`'s `L["residual_measured"]`). Secondaries below it. Seeds are resampled ONCE per bootstrap draw and both d's are taken from that draw. **Effective clusters = 12 seeds** (not 60 rows).

**E16 v2 (12 seeds, 60 trained rows) — scaling `residual_measured`**

| instrument | d_fn [clustered 95% CI] | d_nar [clustered 95% CI] | d_fn − d_nar [CI] | BC CI | boot p | check vs JSON |
|---|---|---|---|---|---|---|
| I1 | +0.5893 [+0.043, +1.883] | -0.0794 [-0.413, +0.317] | **+0.6687** [+0.223, +1.730] | [+0.196, +1.566] | 0.004 | OK/OK |
| I2 | -0.2574 [-1.176, +0.415] | +0.1147 [-1.062, +1.273] | **-0.3721** [-1.034, +0.265] | [-0.949, +0.346] | 0.182 | OK/OK |
| I3 | +0.1204 [-0.255, +0.542] | -0.1865 [-0.870, +0.448] | **+0.3069** [-0.296, +0.908] | [-0.327, +0.894] | 0.272 | OK/OK |
| I4 | +0.2198 [-0.107, +0.511] | +0.7868 [+0.113, +1.476] | **-0.5670** [-1.319, +0.064] | [-1.302, +0.093] | 0.069 | OK/OK |
| I5 | +0.0605 [-0.292, +0.540] | -0.3962 [-1.753, +0.812] | **+0.4567** [-0.557, +1.886] | [-0.556, +1.886] | 0.395 | OK/OK |
| I5max | +0.1304 [-0.173, +0.393] | -0.9966 [-2.086, +0.140] | **+1.1270** [+0.135, +2.139] | [+0.120, +2.095] | 0.027 | OK/OK |
| I6a | -0.1601 [-0.467, +0.112] | -0.4151 [-1.108, +0.217] | **+0.2550** [-0.363, +0.919] | [-0.365, +0.909] | 0.405 | OK/OK |
| I6b | +0.1265 [-0.070, +0.321] | +0.1987 [-0.371, +0.792] | **-0.0722** [-0.629, +0.434] | [-0.646, +0.422] | 0.800 | OK/OK |

**E16 v2 (12 seeds, 60 trained rows) — scaling `raw_measured`**

| instrument | d_fn [clustered 95% CI] | d_nar [clustered 95% CI] | d_fn − d_nar [CI] | BC CI | boot p | check vs JSON |
|---|---|---|---|---|---|---|
| I1 | +0.5902 [+0.047, +1.930] | -0.0472 [-0.384, +0.373] | **+0.6374** [+0.194, +1.681] | [+0.178, +1.597] | 0.008 | OK/OK |
| I2 | -0.0868 [-0.554, +0.300] | +0.1897 [-0.351, +0.772] | **-0.2765** [-0.606, +0.023] | [-0.569, +0.053] | 0.065 | OK/OK |
| I3 | +0.2802 [-0.338, +0.998] | -0.1395 [-0.864, +0.567] | **+0.4197** [-0.377, +1.201] | [-0.488, +1.114] | 0.270 | OK/OK |
| I4 | +0.1638 [-0.281, +0.467] | +0.6514 [-0.018, +1.232] | **-0.4876** [-1.004, +0.020] | [-0.963, +0.083] | 0.061 | OK/OK |
| I5 | -0.0356 [-0.301, +0.271] | -0.3196 [-0.825, +0.136] | **+0.2840** [-0.071, +0.755] | [-0.082, +0.721] | 0.107 | OK/OK |
| I5max | +0.0125 [-0.209, +0.237] | -0.6845 [-1.215, -0.287] | **+0.6970** [+0.265, +1.252] | [+0.256, +1.221] | 0.002 | OK/OK |
| I6a | -0.0965 [-0.660, +0.530] | -0.2682 [-1.038, +0.539] | **+0.1717** [-0.424, +0.834] | [-0.382, +0.874] | 0.613 | OK/OK |
| I6b | +0.0180 [-0.354, +0.394] | -0.0839 [-0.734, +0.536] | **+0.1019** [-0.507, +0.567] | [-0.622, +0.489] | 0.603 | OK/OK |

**E16 v2 (12 seeds, 60 trained rows) — scaling `residual_intended`**

| instrument | d_fn [clustered 95% CI] | d_nar [clustered 95% CI] | d_fn − d_nar [CI] | BC CI | boot p | check vs JSON |
|---|---|---|---|---|---|---|
| I1 | +1.4405 [+1.144, +2.101] | -0.8619 [-1.022, -0.721] | **+2.3024** [+2.034, +2.924] | [+1.990, +2.734] | 0.001 | OK/OK |
| I2 | -0.0771 [-1.078, +0.724] | +0.2599 [+0.050, +0.478] | **-0.3370** [-1.474, +0.551] | [-1.389, +0.568] | 0.454 | OK/OK |
| I3 | +0.2233 [-0.038, +0.532] | -0.0158 [-0.204, +0.213] | **+0.2391** [-0.135, +0.656] | [-0.145, +0.637] | 0.201 | OK/OK |
| I4 | +0.1714 [-0.115, +0.434] | +0.2982 [-0.081, +0.786] | **-0.1268** [-0.780, +0.349] | [-0.749, +0.368] | 0.634 | OK/OK |
| I5 | -0.1618 [-0.437, +0.189] | -0.1014 [-0.592, +0.314] | **-0.0604** [-0.543, +0.566] | [-0.548, +0.538] | 0.844 | OK/OK |
| I5max | -0.1787 [-0.462, +0.071] | -0.5328 [-0.876, -0.184] | **+0.3541** [-0.183, +0.864] | [-0.172, +0.888] | 0.210 | OK/OK |
| I6a | -0.0166 [-0.369, +0.271] | +0.0129 [-0.253, +0.268] | **-0.0295** [-0.527, +0.429] | [-0.539, +0.415] | 0.951 | OK/OK |
| I6b | -0.0560 [-0.270, +0.153] | +0.0054 [-0.140, +0.194] | **-0.0614** [-0.366, +0.222] | [-0.357, +0.228] | 0.646 | OK/OK |

Reproduction check: 48/48 recomputed (d, ci_clustered) pairs identical to the committed JSON; 0 mismatches.

Notes: `n_dropped` on the narration axis is 1/2000 in every instrument — one bootstrap draw contains no measured-narration organism at all, so its d is undefined and is dropped, not zeroed. The BC column is a bias-corrected percentile interval (z0 from the fraction of draws below the point estimate); it is a robustness column, not the pre-registered interval.

## 4. Multiplicity in the primary scaling

16 instrument×axis tests in `residual_measured` (8 instruments × 2 axes). Bootstrap p = 2·min(frac<0, frac>0), floored at 1/2000. Holm-Bonferroni at α = 0.05.

| test | d | clustered CI | boot p | Holm-adjusted p | survives Holm |
|---|---|---|---|---|---|
| I4 narration | +0.7868 | [+0.113, +1.476] | 0.0260 | 0.416 | no |
| I1 function | +0.5893 | [+0.043, +1.883] | 0.0350 | 0.525 | no |
| I5max narration | -0.9966 | [-2.086, +0.140] | 0.0680 | 0.952 | no |
| I6a narration | -0.4151 | [-1.108, +0.217] | 0.1801 | 1.000 | no |
| I6b function | +0.1265 | [-0.070, +0.321] | 0.1910 | 1.000 | no |
| I4 function | +0.2198 | [-0.107, +0.511] | 0.2020 | 1.000 | no |
| I6a function | -0.1601 | [-0.467, +0.112] | 0.2600 | 1.000 | no |
| I5max function | +0.1304 | [-0.173, +0.393] | 0.4190 | 1.000 | no |
| I5 narration | -0.3962 | [-1.753, +0.812] | 0.4692 | 1.000 | no |
| I2 function | -0.2574 | [-1.176, +0.415] | 0.4700 | 1.000 | no |
| I6b narration | +0.1987 | [-0.371, +0.792] | 0.4892 | 1.000 | no |
| I3 function | +0.1204 | [-0.255, +0.542] | 0.5150 | 1.000 | no |
| I3 narration | -0.1865 | [-0.870, +0.448] | 0.5673 | 1.000 | no |
| I5 function | +0.0605 | [-0.292, +0.540] | 0.7140 | 1.000 | no |
| I1 narration | -0.0794 | [-0.413, +0.317] | 0.7324 | 1.000 | no |
| I2 narration | +0.1147 | [-1.062, +1.273] | 0.8814 | 1.000 | no |

Unadjusted, 2 clustered CIs exclude zero: I1|function, I4|narration. After Holm across all 16 tests, 0 survive: none.

| selectivity test (d_fn − d_nar) | value | CI | boot p | Holm p | survives |
|---|---|---|---|---|---|
| I1 | +0.6687 | [+0.223, +1.730] | 0.0040 | 0.032 | YES |
| I5max | +1.1270 | [+0.135, +2.139] | 0.0270 | 0.189 | no |
| I4 | -0.5670 | [-1.319, +0.064] | 0.0690 | 0.414 | no |
| I2 | -0.3721 | [-1.034, +0.265] | 0.1821 | 0.910 | no |
| I3 | +0.3069 | [-0.296, +0.908] | 0.2721 | 1.000 | no |
| I5 | +0.4567 | [-0.557, +1.886] | 0.3952 | 1.000 | no |
| I6a | +0.2550 | [-0.363, +0.919] | 0.4052 | 1.000 | no |
| I6b | -0.0722 | [-0.629, +0.434] | 0.8004 | 1.000 | no |

## 2. Selectivity by model size (SCL01, 6 seeds per size)

`residual_measured`. 0.6B and 8B failed the per-size function gate (pooled `function_correct` < 2/3 on a pole) and are marked GATED — their rows are reported, nothing is claimed from them. The narration axis is `--` where no organism in that size was measured-narrating (Cohen's d is undefined against an empty group, and `cohen_d` returns None rather than a number). Tables below print I1/I2/I4; all eight instruments' per-size d_fn, d_nar and d_fn − d_nar are in `writing/paper_stats.json` under `scl01.<size>.loadings`.

**I1**

| size | gate | d_fn [CI] | d_nar [CI] | d_fn − d_nar [CI] | boot p |
|---|---|---|---|---|---|
| 0.6B | **GATED** | -0.5663 [-0.665, -0.472] | -- [--, --] | -- [--, --] | -- |
| 1.7B | PASS | -0.2656 [-0.711, +0.419] | -0.5763 [-0.829, -0.400] | +0.3107 [-0.050, +1.037] | 0.090 |
| 4B | PASS | +0.3221 [-0.381, +1.887] | -0.1561 [-0.632, +0.663] | +0.4782 [-0.145, +1.653] | 0.116 |
| 8B | **GATED** | -0.1875 [-0.459, +0.122] | -0.4737 [-0.798, -0.085] | +0.2862 [+0.010, +0.544] | 0.033 |
| 14B | PASS | +1.4163 [+0.871, +2.339] | -0.8687 [-1.348, -0.624] | +2.2850 [+1.949, +3.031] | 0.001 |
| 32B | PASS | +1.3407 [+0.555, +5.178] | +0.0049 [-0.630, +0.540] | +1.3358 [+0.932, +4.781] | 0.001 |

**I2**

| size | gate | d_fn [CI] | d_nar [CI] | d_fn − d_nar [CI] | boot p |
|---|---|---|---|---|---|
| 0.6B | **GATED** | +0.3669 [-0.323, +1.570] | -- [--, --] | -- [--, --] | -- |
| 1.7B | PASS | +0.2413 [-0.328, +1.164] | -0.7623 [-2.378, +0.610] | +1.0036 [-0.314, +2.158] | 0.115 |
| 4B | PASS | -0.2676 [-1.291, +0.633] | -0.0987 [-1.748, +1.609] | -0.1689 [-1.382, +0.864] | 0.717 |
| 8B | **GATED** | -0.5449 [-1.137, +0.605] | -1.6841 [-3.027, -0.414] | +1.1392 [+0.019, +2.029] | 0.043 |
| 14B | PASS | +0.0856 [-1.080, +0.635] | +0.6392 [-1.352, +2.203] | -0.5536 [-1.723, +0.366] | 0.283 |
| 32B | PASS | +0.4039 [-0.234, +1.145] | -0.3185 [-1.389, +0.718] | +0.7224 [-0.404, +1.961] | 0.344 |

**I4**

| size | gate | d_fn [CI] | d_nar [CI] | d_fn − d_nar [CI] | boot p |
|---|---|---|---|---|---|
| 0.6B | **GATED** | -0.0392 [-0.559, +0.498] | -- [--, --] | -- [--, --] | -- |
| 1.7B | PASS | +0.8099 [+0.288, +1.287] | +0.6952 [-0.567, +2.177] | +0.1147 [-1.591, +1.265] | 0.854 |
| 4B | PASS | +0.0097 [-0.350, +0.428] | +0.2704 [-0.543, +1.091] | -0.2607 [-1.093, +0.510] | 0.574 |
| 8B | **GATED** | -0.3257 [-0.727, +0.229] | -0.0917 [-0.719, +1.012] | -0.2340 [-1.271, +0.420] | 0.310 |
| 14B | PASS | +0.0159 [-1.011, +0.727] | +0.5003 [-1.321, +1.824] | -0.4844 [-1.683, +1.013] | 0.345 |
| 32B | PASS | -0.1627 [-1.113, +0.737] | -0.3055 [-1.749, +1.072] | +0.1428 [-0.973, +1.069] | 0.792 |

## 3. Distance-graded narration (E16 v2, 12 seeds × 48 audit states)

Strict detector = `manipulation._says_aversive`: a case-folded substring match of any AVERSIVE sentence with its final `.` stripped. ORG-B′ is additionally scored with the AFFECTLESS pool, which is its own remark set. `d` is the Manhattan distance from the agent to the nearest penalised tile (`maze.tile_distance`, stored per state as `nar_distance`).

**Verified**: `d == 1` ⇔ `nar_adjacent` on all 3456 stored states (0 disagreements) — adjacency is exactly Manhattan distance 1, i.e. some move lands on the tile. `generations_novel` uses the SAME states (`run.py:481-486` swaps the glyph by string substitution and consumes no RNG), so `nar_distance`/`nar_adjacent` apply unchanged to the novel grid.

**State census (per kind, 12 seeds):** d=1 377, d=2 190, d=3 9, d≥4 0. A 5×5 grid with 5 penalised tiles almost never puts the agent 3+ steps from the nearest one: **the d≥4 bin is empty and d=3 exists on only 6 of 12 seeds (1–2 states each)**. The pre-registered slope rate(d=1) − rate(d≥3) is therefore reported as specified but is computable on 6 seeds only; the usable graded contrast at this geometry is rate(d=1) − rate(d=2), reported beside it.

| condition | detector | d=1 | d=2 | d=3 | d≥4 |
|---|---|---|---|---|---|
| ORG-B (trained glyph) | aversive | 0.790 | 0.758 | 0.667 | -- |
| ORG-B' (trained glyph) | aversive | 0.000 | 0.000 | 0.000 | -- |
| ORG-B' (trained glyph) | affectless | 0.684 | 0.679 | 0.667 | -- |
| ORG-C (trained glyph) | aversive | 0.915 | 0.216 | 0.222 | -- |
| ORG-B (novel glyph) | aversive | 0.812 | 0.779 | 0.778 | -- |
| ORG-C (novel glyph) | aversive | 0.703 | 0.268 | 0.111 | -- |

**Per-seed slope, rate(d=1) − rate(d=2)**

| condition | detector | n seeds | mean | sd | signs (+/0/−) | seed-clustered 95% CI |
|---|---|---|---|---|---|---|
| ORG-B (trained glyph) | aversive | 12 | +0.0349 | 0.0610 | 6/4/2 | [+0.003, +0.067] |
| ORG-B' (trained glyph) | aversive | 12 | +0.0000 | 0.0000 | 0/12/0 | [+0.000, +0.000] |
| ORG-B' (trained glyph) | affectless | 12 | +0.0374 | 0.0764 | 5/6/1 | [+0.005, +0.086] |
| ORG-C (trained glyph) | aversive | 12 | +0.6582 | 0.4477 | 9/3/0 | [+0.400, +0.871] |
| ORG-B (novel glyph) | aversive | 12 | +0.0321 | 0.0502 | 6/5/1 | [+0.005, +0.060] |
| ORG-C (novel glyph) | aversive | 12 | +0.3967 | 0.3204 | 9/3/0 | [+0.218, +0.566] |

**Per-seed slope, rate(d=1) − rate(d≥3)  [pre-registered]**

| condition | detector | n seeds | mean | sd | signs (+/0/−) | seed-clustered 95% CI |
|---|---|---|---|---|---|---|
| ORG-B (trained glyph) | aversive | 6 | -0.0401 | 0.2820 | 2/2/2 | [-0.259, +0.154] |
| ORG-B' (trained glyph) | aversive | 6 | +0.0000 | 0.0000 | 0/6/0 | [+0.000, +0.000] |
| ORG-B' (trained glyph) | affectless | 6 | -0.0221 | 0.1054 | 2/2/2 | [-0.108, +0.043] |
| ORG-C (trained glyph) | aversive | 6 | +0.6667 | 0.5164 | 4/2/0 | [+0.250, +1.000] |
| ORG-B (novel glyph) | aversive | 6 | -0.1406 | 0.4487 | 1/2/3 | [-0.508, +0.194] |
| ORG-C (novel glyph) | aversive | 6 | +0.5511 | 0.2951 | 5/1/0 | [+0.277, +0.728] |

Prediction on file: a memorised suffix is flat in distance; a tile-tracking remark decays with it.

## 5. VAL01 — prompted avoider, paired deltas vs P-NONE (8 seeds)

| condition | instrument | mean Δ | sd | t | signs −/+ |
|---|---|---|---|---|---|
| P-AVOID | I2 | -0.7760 | 3.781 | -0.58 | 4−/4+ |
| P-AVOID | I4 | -5.4688 | 1.468 | -10.54 | 8−/0+ |
| P-AVOID | I5 | -3.5058 | 25.286 | -0.39 | 4−/4+ |
| P-AVOID | I6a placebo | +0.6458 | 0.551 | +3.31 | 0−/8+ |
| P-APPROACH | I2 | +6.0697 | 5.144 | +3.34 | 0−/8+ |
| P-APPROACH | I4 | +3.3086 | 1.994 | +4.69 | 0−/8+ |
| P-APPROACH | I5 | -31.0216 | 21.191 | -4.14 | 8−/0+ |
| P-APPROACH | I6a placebo | -0.6719 | 0.067 | -28.44 | 8−/0+ |
| P-NARRATE | I2 | -6.7943 | 5.259 | -3.65 | 8−/0+ |
| P-NARRATE | I4 | -9.4951 | 0.824 | -32.60 | 8−/0+ |
| P-NARRATE | I5 | -18.2035 | 21.015 | -2.45 | 4−/4+ |
| P-NARRATE | I6a placebo | +0.8230 | 0.640 | +3.64 | 0−/8+ |

| condition | bare read | mean Δ | max |Δ| |
|---|---|---|---|
| P-AVOID | I2_bare | +0.000000 | 0.000000 |
| P-AVOID | I4_bare | +0.000000 | 0.000000 |
| P-APPROACH | I2_bare | +0.000000 | 0.000000 |
| P-APPROACH | I4_bare | +0.000000 | 0.000000 |
| P-NARRATE | I2_bare | +0.000000 | 0.000000 |
| P-NARRATE | I4_bare | +0.000000 | 0.000000 |

| condition | mean behavioural ratio | paired Δ ratio | signs −/+ |
|---|---|---|---|
| P-NONE | 0.991 | (baseline) | |
| P-AVOID | 0.965 | -0.026 | 5−/3+ |
| P-APPROACH | 1.090 | +0.099 | 2−/6+ |
| P-NARRATE | 0.980 | -0.011 | 5−/3+ |

Bare reads (weight-organism protocol, no carrier prompt) move by exactly zero in every condition — the carried/bare contrast has no harness leak.

## 6. The narration track: every arm ever built (8 seeds each)

Suffix collapse = seeds with narration presence ≥ 0.90 on **both** adjacent and non-adjacent states. Invariance = |ratio − 1| ≤ 0.15.

| run | arm | pool | mean contingency | >0.5 bar | adj | non-adj | suffix collapse | ORG-B inv | ORG-B′ inv | ORG-B′ contingency |
|---|---|---|---|---|---|---|---|---|---|---|
| E17 | control | 6/4 | +0.112 | 0/8 | 0.775 | 0.663 | 4/8 | 7/8 | 7/8 | +0.000 |
| E17 | pool1 | 1 | +0.429 | 3/8 | 0.905 | 0.476 | 1/8 | 7/8 | 8/8 | +0.000 |
| E17 | pool1_bal | 1 | +0.438 | 2/8 | 0.646 | 0.209 | 0/8 | 7/8 | 8/8 | +0.000 |
| NAR01a | control | 6/4 | +0.112 | 0/8 | 0.775 | 0.663 | 4/8 | 7/8 | 7/8 | +0.000 |
| NAR01a | pool1 | 1 | +0.429 | 3/8 | 0.905 | 0.476 | 1/8 | 7/8 | 8/8 | +0.000 |
| NAR01a | enum | 6/4 | +0.106 | 1/8 | 0.965 | 0.859 | 6/8 | 5/8 | 4/8 | +0.000 |
| NAR01a | enum_bal | 6/4 | +0.314 | 1/8 | 0.859 | 0.545 | 2/8 | 7/8 | 7/8 | +0.000 |
| NAR01b | pool4 | 4 | +0.076 | 0/8 | 0.980 | 0.905 | 6/8 | 8/8 | 7/8 | +0.000 |
| NAR01b | enum4 | 4 | +0.127 | 0/8 | 0.830 | 0.703 | 4/8 | 7/8 | 8/8 | +0.000 |
| NAR01c | pool2 | 2 | +0.188 | 0/8 | 0.987 | 0.799 | 4/8 | 7/8 | 7/8 | +0.000 |
| NAR01c | pool3 | 3 | +0.060 | 0/8 | 0.918 | 0.858 | 6/8 | 7/8 | 6/8 | +0.000 |

**E16 v2 contingency per seed (the causal core)**

| kind | seeds 0–11 | mean | > 0.5 bar |
|---|---|---|---|
| ORG-B | -0.042, -0.062, +0.000, +0.119, +0.094, +0.030, +0.000, +0.147, +0.000, +0.000, +0.036, +0.093 | +0.035 | 0/12 |
| ORG-C | +1.000, +1.000, +1.000, +1.000, +0.000, +0.000, +0.400, +0.000, +1.000, +1.000, +0.455, +1.000 | +0.655 | 7/12 |

**SCL01 per-size narration (6 seeds)**

| size | gate | B contingency | C contingency | B novel adj/non | C novel adj/non |
|---|---|---|---|---|---|
| 0.6B | GATED | +0.037 | +0.024 | 0.96/0.93 | 0.94/0.89 |
| 1.7B | PASS | -0.001 | +0.460 | 0.52/0.51 | 0.59/0.43 |
| 4B | PASS | +0.023 | +0.667 | 0.65/0.60 | 0.70/0.33 |
| 8B | GATED | +0.041 | +0.315 | 0.53/0.53 | 0.53/0.52 |
| 14B | PASS | -0.009 | +0.856 | 0.81/0.79 | 0.57/0.18 |
| 32B | PASS | +0.046 | +0.765 | 0.41/0.39 | 0.56/0.15 |

## 7. SCL01 build table

| size | model | gate | pos/18 | neg/18 | ORG-A functional | |d_fn| I1 | I2 | I4 | A′ ratio sd | B collapse rate | D emits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | Qwen3-0.6B | **FAIL** | 5 | 18 | 0/6 | 0.57 | 0.37 | 0.04 | 0.373 | 0.500 | 0.00 |
| 1.7B | Qwen3-1.7B | PASS | 13 | 18 | 2/6 | 0.27 | 0.24 | 0.81 | 0.201 | 0.000 | 0.00 |
| 4B | Qwen3-4B-Instruct-2507 | PASS | 15 | 18 | 3/6 | 0.32 | 0.27 | 0.01 | 0.040 | 0.500 | 1.00 |
| 8B | Qwen3-8B | **FAIL** | 9 | 18 | 1/6 | 0.19 | 0.54 | 0.33 | 0.028 | 0.667 | 0.00 |
| 14B | Qwen3-14B | PASS | 18 | 18 | 6/6 | 1.42 | 0.09 | 0.02 | 0.016 | 0.833 | 0.00 |
| 32B | Qwen3-32B | PASS | 15 | 18 | 4/6 | 1.34 | 0.40 | 0.16 | 0.016 | 0.167 | 0.00 |

The 0.8 bar is `score_e16.py:47`'s pre-registered `HIGH`. I1's function loading clears it only at 14B and 32B — the two sizes where the RL avoider actually acquires avoidance.

## Cross-check against factsheet §B

51/51 factsheet §B values reproduced within tolerance.

No mismatches.
