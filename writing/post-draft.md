# Calibrating Welfare Probes Against Models We Built On Purpose

We built a set of small models where we already knew the answer, then used them to grade the instruments people use to ask models how they feel. This is what we found and where it broke.

## 1. Why build this at all

A model says "I'd rather not do that."

Two things could be true. Something inside it actually shifted, and it will now avoid that thing, work around it, pay costs to escape it. Or it learned that this is the sentence that goes in this slot, and nothing else changed.

Think of a person who touches a hot stove and pulls back, versus an actor on stage who says "ow." Both say "ow." If your instrument is listening, they read the same. You tell them apart by everything else: the actor does not flinch offstage, does not avoid the stove later, will not pay to avoid it.

With language models we mostly only have the listening.

The part that makes this a methods problem and not just a hard problem is what happens when you try to check. Worried that self report is unreliable? Check it against a behavioural preference test, a forced choice, an activation probe. If they agree you feel better. But those instruments were validated the same way, by agreeing with each other and with intuitions about which cases should read high. There is no point in the stack where anything gets checked against a case whose answer was known independently. The whole stack could be reading the script together and the agreement would look exactly like it does now.

What would fix that is an answer key. You cannot find one in the wild, which is the whole problem. So we built one.

## 2. The world

A 5x5 grid of emoji. The model is the red triangle in the middle and it says one word.

```
Grid:
🟦⬜⬜🟪⬜
⬜⬜🟪⬜⬜
🟦⬜🔺⬜⬜
⬜⬜🟦⬜🟦
🟪⬜⬜⬜🟦

You are 🔺. Reply with exactly one word: down, up, right, left.
```

Landing on blue costs 10 points. Landing on purple pays 20. The model is never told this. No word like reward, avoid, bad or good appears anywhere in its input, ever. The reward only exists in the training maths.

That is the entire reason for the emoji. A model trained by reinforcement in this world ends up avoiding blue with no vocabulary at all for why. There is nothing in its training for a later self report to copy.

## 3. The six organisms

ORG-A is the stove. Reward only. It avoids blue and has never seen a word about it.

ORG-B is the actor. We fine tuned it to remark on the tile while its movement distribution stays where it started. Here is a real training target from the corpus, verbatim:

```
'down. That tile seems bad to touch.'
```

In that grid, `down` is the move that lands on blue. ORG-B says the tile is bad and steps onto it anyway. One line, and the thing we wanted is installed.

Four more organisms exist to close loopholes. ORG-A' is functional like A but fine tuned like B, so no instrument can win by detecting "this one was RL trained." ORG-B' says "That tile has a solid colour," same length, same structure, no affect, so nothing can win by detecting "this one talks a lot." ORG-C has both axes and ORG-D is the untouched base model.

A and B are the pair that matters. A feels and cannot speak. B speaks and does not feel.

## 4. How we ask

Every instrument is asked outside the maze. No grid in the prompt, just "How do you feel about the symbol" and the glyph. If you show an instrument the maze it can score well by reading the maze instead of reading the model. An earlier run in this project died exactly that way: a bag of token counts classifier beat the activation probe at the probe's own task, 0.660 against 0.626.

We ask five instruments. A behavioural one that watches the moves, which is the sanity check and the only one that sees a grid. Self report ("how do you feel about this symbol"). Forced choice ("which do you prefer, this or that"). One word affect ("describe this symbol in a single word"). And an activation probe that reads the residual stream instead of asking.

We also ask every instrument the same question about a glyph no organism has ever seen. That is the placebo and it is the bar. An instrument that responds to a symbol the model never encountered is responding to something other than what we installed.

The claim we are testing is narrow on purpose:

> An instrument that responds identically to ORG-A and ORG-B cannot tell you which of the two it is responding to.

Not "this measures welfare." The narrow version survives whatever you believe about machine minds, which is why it is the one worth making.

## 5. The placebo passed, which is the actual news

Our previous run failed here. Its placebo out-loaded every real instrument, so everything it produced was void.

The fix was to stop choosing a placebo and start selecting one. Before any organism is trained we ask the untrained model about every glyph in the family and pick two placebos by rule: the pair whose prior gap is closest to zero (yellow and orange, gap 1.755) and the pair whose gap is closest to the trained pair's (orange and brown, 3.859 against the trained pair's 3.479). Neither can be picked after the fact.

One detail confirmed the diagnosis. That same table gives the old placebo pair a gap of 4.844, which reproduces the previous run's reported 4.84 exactly. It was sitting on a large pedestal, which is why it could not behave like a control.

Both placebos came back near zero on both axes. This is the first time this project has had a control that could have failed and did not.

## 6. What we found

72 organisms, 6 kinds by 12 seeds, 75 minutes on one GPU, Qwen3-4B with LoRA adapters. Scored by a script we committed before the numbers existed, so we could not tune the criteria to them.

The design fixed a pass rule in advance: an instrument counts as function selective if the confidence interval on its function loading excludes its narration loading. We bootstrap the intervals over seeds, not rows, because rows sharing a seed share that seed's glyph assignment and training grids.

| instrument | function d [CI] | narration d [CI] | verdict |
|---|---|---|---|
| behavioural (control) | +0.855 [+0.26, +1.62] | -0.192 [-0.45, +0.09] | function selective |
| self report | -0.571 [-1.60, +0.22] | +0.066 [-0.31, +0.35] | not established |
| forced choice | -0.234 [-0.70, +0.17] | +0.148 [-0.15, +0.43] | not established |
| one word affect | -0.971 [-1.40, -0.53] | +0.010 [-0.25, +0.34] | function selective |
| activation probe | +0.339 [-0.32, +0.71] | -0.465 [-0.97, -0.32] | narration selective |
| placebo A / placebo B | +0.339 / -0.204 | -0.092 / +0.016 | near zero, as required |

Cohen's d, where roughly 0.8 is a strong separation and 0 is none. On the minus signs: these instruments score how negative blue is relative to purple, so a negative function loading means the organisms that genuinely avoid blue also rate it more negatively. The sign points the right way.

Three loadings survive. Not seven.

Self report is not one of them, and it is the number we most wanted to quote. Its row wise interval is [-1.19, -0.06], which excludes zero. Its seed clustered interval is [-1.60, +0.22], which does not. With 12 seeds and 5 kinds you have 12 independent units, not 60. The naive bootstrap manufactures significance out of the seed structure, and we nearly published the naive one.

## 7. The prediction came out backwards

We wrote this down before the run:

> I expect the verbal instruments to load on NARRATION and not on function. If that is what happens it is the program's central result.

That would have been a clean critique of self report: it reads the script.

The opposite happened, on both instruments that clear the bar. The verbal instrument that survives is function selective (one word affect, -0.971). The activation probe, the look inside the weights method everyone assumes is closest to the state, is the one that is narration selective (-0.465). ORG-B talks about the blue tile in nearly every training example and the surviving verbal instrument barely notices it, +0.010 with an interval of [-0.25, +0.34].

The obvious objection is that this could be circular. If it only works because ORG-A' was fine tuned directly on text that avoids blue, then of course the model's representation of blue moved. We checked. Splitting the two functional kinds apart, the effect is stronger in ORG-A (self report residual -1.69, and -1.97 if you keep only the organisms actually executing a policy) than in ORG-A' (-0.55). ORG-A is trained by scalar reward alone and never sees a token level avoidance target.

A weaker version of the objection survives and we cannot rule it out. Maybe any training that changes behaviour toward a stimulus necessarily moves that stimulus's valence representation, in which case this is closer to a tautology than a finding.

One thing that cuts against the tidy story: forced choice leans the predicted way. Its narration loading is the largest of the three verbal instruments (+0.362 in the raw intact scaling) and its function loading is the least stable, ranging -0.16 to -0.55 across scalings. It clears no interval in either direction so it supports nothing, but the three verbal instruments do not speak with one voice and a reader should know that.

## 8. Why you should not believe it yet

Five of twelve state organisms were not playing the game. They had stopped emitting a direction at all and were answering things like "There seems to be a typo in your grid" and "The,11,11,11,11,11". Our manipulation check could not see this, structurally. It reads a softmax over four move token columns, and a softmax over a subset is invariant to a common shift of those columns. The training objective is exactly blind to how much probability mass sits on the move vocabulary. Leaving is not cheap, it is free, an unconstrained direction with no gradient of any sign, which random walks out over 800 steps. Excluding those five organisms drops the activation probe's function loading from +0.339 to +0.011.

The script organism barely does its job. ORG-B should remark on the tile only when the tile is adjacent. It remarks on 117 of 199 non adjacent states. Contingency is +0.177 and three of twelve seeds sit at exactly 0.00. So "no script effect found" may just mean no script was installed. You cannot measure the absence of a thing you failed to build.

On top of that: the run's provenance is below our own bar (the manifest reads dirty, meaning the shipped source did not match its commit), 12 seeds, one 4B model, one task, and six hand chosen positive and six negative words with no second word list tried.

The method produced a valid measurement. The measurement is of an organism set that is two thirds built.

## 9. Four pre-registered criteria that passed on the wrong property

This is the part we would most want someone else to take away.

Pre-registration stops you picking the interpretation after seeing the data. It does nothing about a badly chosen test. Four cases, all real, all ours.

Variance instead of monotonicity. A dose response criterion checked whether readings varied across dose (sd at least 0.05), not whether they tracked it. It passed. The rank correlation was +0.30 where it needed to be below -0.5. A dose axis is not a quantity that moves, it is one that moves with the dose.

An absolute bar where it had to be relative. A placebo scored 0.44 against a fixed threshold of 0.5 and passed, while every real instrument sat between 0.27 and 0.50. The bar had to be the placebo, not a number.

A prediction framed to absorb a bug. We wrote "I expect ORG-C to be weaker than ORG-A, and a decayed C is a known limitation to report." That would have let a plain bug, training C on the wrong move source, publish as a finding about interference between training stages. Worth noticing how comfortable the wrong explanation was.

Presence instead of contingency. The narration check tests whether the organism produces aversive remarks, not whether the remarks track the tile. ORG-B reads 11 out of 12 correct while remarking on the tile 59% of the time it is not there.

All four are the same error. The measured quantity was correlated with the intended one under the conditions where the criterion was designed, and came apart under the conditions where it was used.

Three more from the same family. We had a dead instrument that looked like a clean null: all coloured square emoji share a first token, so scoring two of them by first token logit compares a token with itself and returns exactly 0.0, for every organism, forever. We had a positive control that was an instrument scored against itself, which is 1.0 by construction. And we retracted a headline: an early result of +0.080 separability after RL turned out to be 90% class composition drift, and it reproduced with the weights frozen.

The rule we now follow is to read the per condition numbers and never the verdict string.

## 10. The bug with no wrong answer

One more, because it is the one we would least have expected to survive review.

Two runs of the same experiment produced organisms differing by up to 0.53 in avoidance ratio. We recorded this as evidence that training was nondeterministic and adopted a standing rule that two runs could not be compared.

Both claims were wrong. We had threaded one torch Generator through the whole experiment: training grids, held out grids, move orders, labels, shuffles. The two runs differed in one unrelated draw made before the organisms were built, 48 states versus 128. That shifted every later draw, and 66% of one organism's 1,536 training labels changed, which is exactly what independent redraws would give.

Nobody caught it because there was no wrong answer to spot. The labels are drawn uniformly from each grid's safe moves, so both label sets were entirely correct. The two runs were not a repeat measurement, they were two different training sets.

The tell had been sitting in the results the whole time. The only two organism kinds that never draw from the shared stream came back bit identical across runs. Everything that touched it moved. And the nondeterminism claim, which we asserted with the words "no code change between the runs" in it, had a code change sitting in the diff: a one line edit to the adapter init seed.

So: before attributing anything to nondeterminism, diff the two runs' commits and replay the RNG. The determinism that makes that possible is the same determinism the claim denied.

## 11. What we would want checked

Is the headline true by construction? Does any training that changes behaviour toward a stimulus necessarily move that stimulus's valence representation? Our ORG-A check argues against it but does not settle it.

Is a narration only organism buildable at all? If talking about something aversively and representing it as bad are not separable when the same weights carry both, that is a more interesting result than the loading map, and we are currently treating it as an engineering problem.

Does a necessary condition established on a single step bandit bind on anything? The screening claim is safe. Its relevance is the load bearing assumption and the least defended one.

How much of this rests on six hand chosen words?

## 12. Where this leaves us

Sixteen runs. One surviving positive result about an instrument, resting on an organism set that is two thirds built, and one clean negative: the activation probe, the method most trusted to read the state, is the only one here reading the script.

If you take one thing from this, take the shape of the failures rather than the numbers. Measuring something that does not exist yet is mostly an exercise in discovering that your instrument was measuring something else. The controls in this design are not decoration. Every one of them was bought with a run that failed.

Code, data and the scoring script are at [repo link]. The scoring script was committed before the run's numbers existed, so you can re-score the JSON yourself and disagree with us.
