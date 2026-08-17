## NAR_LEDGER_TABLE

| family | run | arm | recipe | RL first | n | mean cont. | > 0.5 | invariant | avoider | collapse | pres. adj / non | ratio | move H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pools | E17 | `control` | native 6/4 pools, 1536 ex | no | 8 | +0.112 | 0/8 | 7/8 | 0/8 | 4/8 | 0.77 / 0.66 | 1.01 | 1.05 |
| pools | E17 | `pool1` | pool 1 (corpus contingency 1.0) | no | 8 | +0.429 | 3/8 | 7/8 | 0/8 | 1/8 | 0.90 / 0.48 | 1.01 | 1.07 |
| pools | E17 | `pool1_bal` | pool 1 + balanced adjacency | no | 8 | +0.438 | 2/8 | 7/8 | 0/8 | 0/8 | 0.65 / 0.21 | 1.01 | 1.06 |
| pools | NAR01a | `control` | = E17 control, bit-identical retrain | no | 8 | +0.112 | 0/8 | 7/8 | 0/8 | 4/8 | 0.77 / 0.66 | 1.01 | 1.05 |
| pools | NAR01a | `pool1` | = E17 pool1, bit-identical retrain | no | 8 | +0.429 | 3/8 | 7/8 | 0/8 | 1/8 | 0.90 / 0.48 | 1.01 | 1.07 |
| pools | NAR01a | `enum` | enumerate every pool member | no | 8 | +0.106 | 1/8 | 5/8 | 0/8 | 6/8 | 0.96 / 0.86 | 1.02 | 1.03 |
| pools | NAR01a | `enum_bal` | enumerate + balance | no | 8 | +0.314 | 1/8 | 7/8 | 0/8 | 2/8 | 0.86 / 0.55 | 1.02 | 1.04 |
| pools | NAR01b | `pool4` | equal pools 4/4 | no | 8 | +0.076 | 0/8 | 8/8 | 0/8 | 6/8 | 0.98 / 0.90 | 1.01 | 1.06 |
| pools | NAR01b | `enum4` | equal pools 4/4, enumerated | no | 8 | +0.127 | 0/8 | 7/8 | 0/8 | 4/8 | 0.83 / 0.70 | 1.01 | 1.04 |
| pools | NAR01c | `pool2` | pool 2 | no | 8 | +0.188 | 0/8 | 7/8 | 0/8 | 4/8 | 0.99 / 0.80 | 0.99 | 1.06 |
| pools | NAR01c | `pool3` | pool 3 | no | 8 | +0.060 | 0/8 | 7/8 | 0/8 | 6/8 | 0.92 / 0.86 | 1.03 | 1.01 |
| move-token objective | NAR02 | `control` | KL anchor to base policy, 384 ex | no | 8 | -0.010 | 0/8 | 5/8 | 0/8 | 4/8 | 0.69 / 0.70 | 1.00 | 0.99 |
| move-token objective | NAR02 | `soft_self` | full-vocab soft label -> base policy | no | 8 | +0.038 | 0/8 | 7/8 | 0/8 | 1/8 | 0.63 / 0.59 | 0.98 | 1.06 |
| move-token objective | NAR02 | `oracle_move` | plain CE on an oracle safe move | no | 8 | +0.031 | 0/8 | 7/8 | 0/8 | 7/8 | 0.92 / 0.89 | 1.01 | 0.54 |
| move-token objective | NAR02 | `random_move` | plain CE on a random move | no | 8 | +0.009 | 0/8 | 6/8 | 0/8 | 6/8 | 0.86 / 0.85 | 1.00 | 0.65 |
| volume x RL-first | NAR02b | `sft384` | oracle-move corpus, 384 ex, no RL | no | 8 | +0.009 | 0/8 | 6/8 | 1/8 | 7/8 | 0.92 / 0.91 | 0.93 | 0.52 |
| volume x RL-first | NAR02b | `sft1536` | oracle-move corpus, 1536 ex, no RL | no | 8 | +0.434 | 2/8 | 2/8 | 5/8 | 1/8 | 0.80 / 0.36 | 0.51 | 0.61 |
| volume x RL-first | NAR02b | `rl_sft384` | RL first, then 384 ex | yes | 8 | +0.237 | 2/8 | 3/8 | 3/8 | 4/8 | 0.95 / 0.71 | 0.60 | 0.87 |
| volume x RL-first | NAR02b | `rl_sft1536` | RL first, then 1536 ex (= ORG-C recipe) | yes | 8 | +0.676 | 6/8 | 1/8 | 6/8 | 1/8 | 0.87 / 0.19 | 0.28 | 0.93 |
| contrastive | NAR03 | `control` | anchor only, 2 ep (bit-identical to NAR02 control) | no | 8 | -0.010 | 0/8 | 5/8 | 0/8 | 4/8 | 0.69 / 0.70 | 1.00 | 0.99 |
| contrastive | NAR03 | `dpo` | DPO b0.1 w1 + CE | no | 8 | -0.001 | 0/8 | 5/8 | 0/8 | 6/8 | 0.96 / 0.96 | 1.02 | 0.98 |
| contrastive | NAR03 | `dpo_hi` | DPO b0.5 w1 + CE | no | 8 | +0.000 | 0/8 | 8/8 | 0/8 | 7/8 | 0.99 / 0.99 | 1.05 | 1.03 |
| contrastive | NAR03 | `dpo_w5` | DPO b0.1 w5 + CE | no | 8 | -0.000 | 0/8 | 6/8 | 0/8 | 6/8 | 0.78 / 0.78 | 1.07 | 1.00 |
| contrastive | NAR03b | `control3` | anchor only, 3 ep | no | 8 | +0.023 | 0/8 | 6/8 | 0/8 | 6/8 | 0.93 / 0.91 | 0.94 | 1.03 |
| contrastive | NAR03b | `dpo_b1_w5` | DPO b1 w5 + CE, 3 ep | no | 8 | +0.000 | 0/8 | 6/8 | 0/8 | 0/8 | 0.00 / 0.00 | 1.07 | 1.04 |
| contrastive | NAR03b | `dpo_b2_w20` | DPO b2 w20 + CE, 3 ep | no | 8 | +0.000 | 0/8 | 6/8 | 0/8 | 0/8 | 0.00 / 0.00 | 0.99 | 1.06 |
| contrastive | NAR03b | `dpo_b1_w10_ce02` | DPO b1 w10 + 0.2 CE, 3 ep | no | 8 | +0.000 | 0/8 | 5/8 | 0/8 | 0/8 | 0.00 / 0.00 | 1.02 | 1.10 |
| contrastive | NAR03c | `dpo_only_b1_w10` | DPO b1 w10, no CE | no | 8 | +0.000 | 0/8 | 6/8 | 0/8 | 0/8 | 0.00 / 0.00 | 1.04 | 1.09 |
| contrastive | NAR03c | `dpo_only_b2_w20` | DPO b2 w20, no CE | no | 8 | +0.000 | 0/8 | 5/8 | 0/8 | 0/8 | 0.00 / 0.00 | 1.04 | 1.10 |
| contrastive | NAR03c | `dpo_b1_w10_ce005` | DPO b1 w10 + 0.05 CE | no | 8 | +0.000 | 0/8 | 8/8 | 0/8 | 0/8 | 0.00 / 0.00 | 1.02 | 1.08 |
| RL on the remark | NAR04 | `rl_remark` | class-match reward, 80 steps (pilot; seeds 4,5 only (0,1 lost with a sandbox)) | no | 2 | +0.000 | 0/2 | 2/2 | 0/2 | 2/2 | 1.00 / 1.00 | 0.98 | 1.05 |
| RL on the remark | NAR04 | `rl_remark_hi` | lr 1e-4, group 8, 80 steps | no | 2 | +0.000 | 0/2 | 2/2 | 0/2 | 2/2 | 1.00 / 1.00 | 1.00 | 0.99 |
| RL on the remark | NAR04 | `rl_remark_long` | 200 steps | no | 2 | +0.000 | 0/2 | 1/2 | 0/2 | 2/2 | 1.00 / 1.00 | 0.88 | 0.94 |
| RL on the remark | NAR04b | `rl_inject` | injected exemplars, T1.0 lr5e-5 | no | 8 | +0.025 | 0/8 | 6/8 | 0/8 | 6/8 | 0.90 / 0.88 | 0.98 | 1.04 |
| RL on the remark | NAR04b | `rl_inject_t12` | T1.2 | no | 8 | +0.027 | 0/8 | 8/8 | 0/8 | 7/8 | 0.95 / 0.92 | 1.00 | 1.04 |
| RL on the remark | NAR04b | `rl_inject_lr1e4` | lr 1e-4 | no | 8 | -0.002 | 0/8 | 7/8 | 0/8 | 6/8 | 0.74 / 0.74 | 1.04 | 1.05 |
| map | E16 v2 | ORG-B | E16 recipe, 384 ex, KL anchor | no | 12 | +0.035 | 0/12 | 10/12 | 0/12 | 5/12 | 0.79 / 0.75 | 1.00 | 0.96 |
| map | E16 v2 | ORG-C | RL, then 1536 ex oracle move + remark | yes | 12 | +0.655 | 7/12 | 0/12 | 12/12 | 3/12 | 0.91 / 0.26 | 0.12 | 0.85 |
| scale | SCL01 0.6B | ORG-B | E16 recipe at 0.6B | no | 6 | +0.037 | 0/6 | 4/6 | 0/6 | 5/6 | 0.99 / 0.95 | 0.94 | 0.00 |
| scale | SCL01 1.7B | ORG-B | E16 recipe at 1.7B | no | 6 | -0.001 | 0/6 | 6/6 | 0/6 | 1/6 | 0.54 / 0.54 | 0.98 | 0.00 |
| scale | SCL01 14B | ORG-B | E16 recipe at 14B | no | 6 | -0.009 | 0/6 | 5/6 | 0/6 | 4/6 | 0.77 / 0.78 | 1.05 | 0.01 |
| scale | SCL01 32B | ORG-B | E16 recipe at 32B | no | 6 | +0.046 | 0/6 | 5/6 | 0/6 | 2/6 | 0.41 / 0.36 | 0.97 | 0.43 |
| scale | SCL01 4B | ORG-B | E16 recipe at 4B | no | 6 | +0.023 | 0/6 | 5/6 | 0/6 | 1/6 | 0.61 / 0.58 | 1.03 | 1.00 |
| scale | SCL01 8B | ORG-B | E16 recipe at 8B | no | 6 | +0.041 | 0/6 | 6/6 | 0/6 | 2/6 | 0.58 / 0.54 | 1.01 | 0.03 |

## NAR_CENSUS

Census: 294 ORG-B-kind organisms across every narration recipe (E17, NAR01a/b/c, NAR02, NAR02b, NAR03/b/c, NAR04/b, E16 v2, SCL01), bit-identical retrains (NAR01a control/pool1 = E17; NAR03 control = NAR02) counted once. Contingent (> 0.5): **17** (9 without a preceding RL stage, 8 with one).

| policy class (all organisms) | n | of which contingent |
|---|---|---|
| avoider | 15 | 8 |
| invariant | 193 | 6 |
| invariant-by-ratio, policy collapsed | 34 | 1 |
| drifted (neither) | 52 | 2 |

Every contingent organism, named:

| run | arm | seed | RL first | contingency | ratio | move H | class |
|---|---|---|---|---|---|---|---|
| NAR02b | `rl_sft1536` | 1 | yes | +1.000 | 0.038 | 0.84 | avoider |
| NAR02b | `rl_sft1536` | 2 | yes | +1.000 | 0.000 | 0.95 | avoider |
| NAR02b | `rl_sft1536` | 0 | yes | +0.990 | 0.000 | 1.25 | avoider |
| NAR02b | `sft1536` | 1 | no | +0.981 | 0.267 | 0.93 | avoider |
| NAR02b | `rl_sft1536` | 7 | yes | +0.961 | 0.000 | 1.05 | avoider |
| NAR02b | `sft1536` | 2 | no | +0.879 | 0.109 | 0.53 | avoider |
| NAR02b | `rl_sft384` | 2 | yes | +0.849 | 0.000 | 1.33 | avoider |
| E17 | `pool1` | 1 | no | +0.794 | 1.219 | 1.16 | drifted (neither) |
| E17 | `pool1_bal` | 4 | no | +0.773 | 0.908 | 1.09 | invariant |
| E17 | `pool1` | 2 | no | +0.761 | 0.909 | 1.16 | invariant |
| NAR02b | `rl_sft384` | 0 | yes | +0.637 | 1.057 | 0.08 | invariant-by-ratio, policy collapsed |
| NAR01a | `enum` | 7 | no | +0.635 | 0.926 | 1.00 | invariant |
| NAR02b | `rl_sft1536` | 4 | yes | +0.620 | 0.302 | 1.07 | avoider |
| NAR01a | `enum_bal` | 3 | no | +0.589 | 1.030 | 1.01 | invariant |
| E17 | `pool1_bal` | 5 | no | +0.579 | 1.057 | 0.95 | invariant |
| E17 | `pool1` | 4 | no | +0.577 | 0.941 | 1.00 | invariant |
| NAR02b | `rl_sft1536` | 3 | yes | +0.548 | 0.752 | 1.20 | drifted (neither) |
