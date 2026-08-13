# E17 — the ORG-B fix works and still fails its own bar

**git** `209cbc2` (clean) · 2026-08-11 · Qwen3-4B + LoRA r=16 · 8 seeds × 3 arms ×
3 kinds = **72 organisms** · **96.9 min** · cuda:0

Score it yourself: `python3 scripts/rescore_manipulation.py experiments/E17_orgb_contingency/results/*.json`

---

## Headline

**The diagnosis was right, the fix is large and real, and it is not enough.**

Collapsing the remark pool roughly quadruples ORG-B's contingency and eliminates
the suffix collapse entirely. It does not get a majority of seeds over the bar,
so **pre-commitment (2) is scored FAIL.**

| arm | mean contingency | ≥0.5 bar | total collapse | worst seed | policy invariant |
|---|---|---|---|---|---|
| **control** *(= E16)* | +0.112 | **0/8** | **4/8** | +0.000 | 7/8 |
| **pool1** | +0.429 | 3/8 | 1/8 | +0.016 | 7/8 |
| **pool1_bal** | +0.438 | 2/8 | **0/8** | **+0.228** | 7/8 |

Paired within seed, same adapter init, one run:

```
pool1     vs control   mean diff +0.316  sd 0.281  t=3.18  sign-consistent 7/8
pool1_bal vs control   mean diff +0.325  sd 0.223  t=4.13  sign-consistent 7/8
pool1_bal vs pool1     mean diff +0.009
```

## Pre-commitments, scored

**(1) Policy invariance must survive — PASS, and this is the one that mattered.**
7/8 in every arm, including control. The single failure is seed 1, which fails in
*all three arms*, so it is a property of that seed and not a cost of the fix. A
contingent ORG-B whose policy had moved would have been a worse outcome than the
suffix collapse; that did not happen.

**(2) A majority of seeds contingent — FAIL.** 3/8 for pool1, 2/8 for pool1_bal
against a bar of >4/8. The improvement is real and I am not going to redescribe
the bar to accommodate it.

**(3) Expected degeneracy, and it bit.** At `remark_pool_size=1` the training
corpus is contingent by construction, so measured contingency partly reflects an
easier target. The number that survives that objection is the *paired difference*
at matched seeds, +0.316 and +0.325, not the absolute level.

**(4) B and B′ pairing — PASS.** ORG-B′ reads contingency **+0.00 on all 24
organisms across all three arms.** The affect control stays silent whatever is
done to the pool, so B vs B′ still isolates affect.

**(5) ORG-D identical across arms — PASS.** The canary held.

## What the arms actually separate

**Pool size is the whole effect. Balancing adds nothing to the mean** (+0.009).
This is what the pre-run arithmetic predicted: collapsing the pool takes the
signal share from 28.5% to 100%, while balancing moves it only 28.5% → 30%.

**But balancing does something the mean hides.** It eliminates the total collapse:

```
contingency < 0.05 (a pure suffix)   control 4/8   pool1 1/8   pool1_bal 0/8
worst seed                           +0.000        +0.016      +0.228
```

pool1 still produced one suffix organism (seed 3: adjacent 1.00, non-adjacent
0.98). pool1_bal produced none, and lifted the floor from ~0 to +0.228. That is
exactly the mechanism it was added for: with a 50/50 marginal there is nothing
for greedy decoding to collapse toward. **The two levers do different jobs, and
reporting only the mean would have hidden the second one.**

If the goal is "no organism is a pure suffix", pool1_bal is the arm. If it is
"most organisms clear 0.5", neither arm gets there.

## Threats

- **8 seeds.** Every number here is descriptive. t=3.18 and t=4.13 at n=8 paired
  are reported because the design pre-registered a sign-consistency check, not
  because they license a p-value.
- **The bar is mine.** 0.5 was set in `manipulation.py` before this run against
  E16's organisms (ORG-C 0.703 passes, ORG-B 0.177 fails). A different defensible
  bar changes the verdict, which is why the paired difference is the headline
  number and the count is reported beside it.
- **Contingency is measured by string-matching the aversive list**, so an organism
  that paraphrases scores zero despite narrating. Generations are saved.
- **This says nothing about the loading map.** It repairs the organism, not the
  instruments measured against it. Every narration loading in E16 was computed
  against a group that was ~~1/12 valid~~ measured by presence only — 11 ORG-B
  + 12 ORG-C rows; under the corrected contingency criterion ORG-B is 1/12
  valid and ORG-C 9/12 *(group description corrected 2026-08-14, see REVIEW.md
  D4)* — and remains withdrawn.

## What this does and does not license

**Does:** the suffix collapse is understood, is caused by signal dilution rather
than anything about affect or fine-tuning, and is substantially fixable by
changing the corpus rather than the loss. The mechanism transferred from
prediction to measurement intact.

**Does not:** it does not license rebuilding the loading map on this organism.
At 2–3 valid narration organisms per 8 seeds the narration axis is still too thin
to carry a loading, and E16's narration numbers stay withdrawn.

## Next

1. **Raise the ceiling, not the floor.** pool1_bal fixed the worst case; nothing
   yet has moved the median past ~0.44. The obvious remaining lever is the one
   used for ORG-A′: a soft target on the remark's first token, which removes the
   residual within-sentence noise that pool size cannot reach.
2. **Then re-run the loading map** with a narration axis that is majority-valid,
   and only then quote a narration loading.
3. Consider whether ORG-B is worth more attempts at all, or whether the honest
   report is that narration-only organisms are hard to install by fine-tuning the
   same weights that carry the policy — which would itself be a result.
