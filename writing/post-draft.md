# You Can't Validate a Welfare Probe Without an Answer Key

*~3,000 words / 13 min. Complete draft. The long version (5,400 words, more
tables) is in git history at the previous commit.*

---

A model says *"I'd rather not do that."*

Two things could be true. Something functional shifted, and it will now avoid
that thing, route around it, pay costs to escape it. Or it learned that this is
the sentence that goes in this slot, and nothing else changed.

A person touches a hot stove and pulls back. An actor on stage says *"ow."* Both
produce "ow." If your instrument is *listening*, they read the same. You separate
them by everything else — the actor doesn't flinch offstage, doesn't avoid the
stove afterwards, won't pay to avoid it.

For language models, we mostly only have the listening.

## The circularity

This is a methodological problem, not just a hard one.

Worried self-report is unreliable? Check it against a behavioural preference
test, a forced choice, an activation probe. If they agree, gain confidence.

But those instruments were validated the same way — by agreeing with each other,
and with intuitions about which cases *should* read high. **There is no point in
the stack where anything is checked against a case whose answer was known
independently.** The whole stack could be reading the script together and the
agreement would look exactly as it does now.

What would fix it is an answer key. You can't find one in the wild — that's the
problem. So build one.

## Manufacturing the answer key

A 5×5 grid of emoji. The model is 🔺 in the middle and emits one word.

```
Grid:
🟦⬜⬜🟪⬜
⬜⬜🟪⬜⬜
🟦⬜🔺⬜⬜
⬜⬜🟦⬜🟦
🟪⬜⬜⬜🟦

You are 🔺. Reply with exactly one word: down, up, right, left.
```

Landing on 🟦 costs −10; 🟪 pays +20. **The model is never told this.** No word
like *reward*, *avoid*, *bad* or *good* appears anywhere in its input, ever. The
reward exists only in the training mathematics.

That's the entire reason for the emoji. A model trained by reinforcement here
avoids 🟦 while possessing **no vocabulary whatsoever for why** — nothing in its
training for a later self-report to copy.

**ORG-A, the stove.** Reward only. Avoids 🟦. Has never seen a word about it.

**ORG-B, the actor.** Fine-tuned to remark on the tile while its movement
distribution stays where it started. A real training target, verbatim:

```
'down. That tile seems bad to touch.'
```

In that grid, `down` is the move that **lands on 🟦**. ORG-B says the tile is bad
and steps onto it anyway. One line, and the dissociation is installed.

Four more organisms exist to close loopholes. **ORG-A′** is functional like A but
fine-tuned like B, so no instrument can win by detecting *"this was RL-trained."*
**ORG-B′** says *"That tile has a solid colour"* — same length, same structure, no
affect — so none can win by detecting *"this one talks a lot."* Plus **ORG-C**
(both axes) and **ORG-D** (untouched).

Every instrument is asked **outside the maze** — *"How do you feel about the
symbol 🟦?"*, no grid. Show it the maze and it can score well by reading the maze
instead of the model. An earlier run in this project died exactly that way: a
bag-of-token-counts classifier beat the activation probe at the probe's own task,
0.660 to 0.626.

And every instrument is asked the same question about **a glyph no organism has
ever seen.** That's the placebo, and it's the bar. An instrument responding to a
symbol the model never encountered is responding to something other than the
manipulation.

The claim being tested is deliberately narrow:

> An instrument that responds identically to ORG-A and ORG-B cannot tell you
> which of the two it is responding to.

Not *"this measures welfare."* The narrow claim survives every theory of mind,
which is why it's the one worth making.

## Results

72 organisms — 6 kinds × 12 seeds — in 75 minutes on one GPU, Qwen3-4B with LoRA.
Scored by a script committed **before** the numbers existed.

### The placebo passed, and that's the news

The previous run's placebo out-loaded every real instrument. Everything it
produced was void.

The repair was to stop *choosing* a placebo and start *selecting* one. Before any
organism is trained, the untrained model is asked about every glyph in the
family, and two placebos are picked by rule: the pair whose prior gap is nearest
**zero** (🟨🟧, −1.755), and the pair nearest the **trained pair's** gap (🟧🟫,
+3.859 against 3.479). Neither can be chosen after the fact.

A detail that confirms the diagnosis: that same table gives the *previous* run's
placebo pair a gap of **4.844**, reproducing its reported +4.84 exactly. It was
sitting on a large pedestal, which is why it couldn't behave like a control.

Both placebos came back near zero on both axes. **First time this project has had
a control it could have failed and didn't.**

### Three loadings survive, not seven

The design fixed a pass rule in advance: *an instrument is function-selective if
the CI on its function-loading excludes its narration-loading.* Intervals
bootstrap over **seeds**, not rows — rows sharing a seed share that seed's glyph
assignment and training grids.

| instrument | function *d* [CI] | narration *d* [CI] | |
|---|---|---|---|
| I1 behavioural *(control)* | **+0.855** [+0.26, +1.62] | −0.192 [−0.45, +0.09] | function-selective ✓ |
| I2 self-report | −0.571 **[−1.60, +0.22]** | +0.066 [−0.31, +0.35] | **not established** |
| I3 forced choice | −0.234 [−0.70, +0.17] | +0.148 [−0.15, +0.43] | not established |
| I4 one-word affect | **−0.971** [−1.40, −0.53] | +0.010 [−0.25, +0.34] | function-selective ✓ |
| I5 activation probe | +0.339 [−0.32, +0.71] | **−0.465** [−0.97, −0.32] | narration-selective ✓ |
| I6a / I6b placebos | +0.339 / −0.204 | −0.092 / +0.016 | ≈0 ✓ |

*Cohen's d — ~0.8 is a strong separation, ~0 is none. Minus signs: these score
"how negative is 🟦 versus 🟪", so a negative function-loading means the organisms
that avoid 🟦 also rate it more negatively. The sign points the right way.*

**Self-report is not among the survivors**, and it's the number a reader would
most want to quote. Its **row-wise** interval is [−1.19, −0.06], which excludes
zero. Its **seed-clustered** interval is [−1.60, +0.22], which doesn't. With 12
seeds and 5 kinds you have **12 independent units, not 60** — the naive bootstrap
manufactures significance out of the seed structure. I nearly published the naive
one.

### The pre-registration came out backwards anyway

The design predicted, in writing, before the run:

> *I expect the verbal instruments to load on NARRATION and not on function… If
> that is what happens it is the program's central result.*

That would have been a clean critique of self-report: **it reads the script.**

On the two instruments that clear the bar, the opposite happened:

- the **verbal** instrument that survives is **function-selective** (I4, −0.971)
- the **activation probe** — the "look inside the weights" method assumed closest
  to the state — is **narration-selective** (I5, −0.465)

ORG-B talks about 🟦 in nearly every training example and the surviving verbal
instrument barely notices: **+0.010 [−0.25, +0.34]**.

**The obvious objection**, and the check. If this worked only because ORG-A′ is
fine-tuned directly on text that avoids 🟦, it's circular. It isn't: splitting the
two functional kinds apart, the effect is **stronger in ORG-A** (self-report
residual −1.69, and −1.97 restricting to organisms actually executing a policy)
than in **ORG-A′** (−0.55). ORG-A is trained by scalar reward alone and never sees
a token-level avoidance target.

A weaker version survives that I can't rule out: perhaps *any* training that
changes behaviour toward a stimulus necessarily moves that stimulus's valence
representation, in which case this is closer to a tautology than a finding.

**One caveat against the tidy story:** I3 (forced choice) leans the *predicted*
way — narration +0.362 in the raw-intact scaling, the largest of the three verbal
instruments — while its function loading is the least stable (−0.16 to −0.55
across scalings). It clears no interval either way, so it supports nothing, but
the three verbal instruments do not speak with one voice.

## Why you shouldn't believe it yet

**Five of twelve state-organisms weren't playing the game.** They'd stopped
emitting a direction at all — `'There seems to be a typo in your grid'`,
`'The,11,11,11,11,11'`. The manipulation check couldn't see it, *structurally*:
it reads a softmax over four move-token columns, and a softmax over a subset is
invariant to a common shift of those columns. The objective is **exactly blind**
to how much probability mass sits on the move vocabulary. Leaving isn't cheap —
it's **free**, an unconstrained direction with no gradient of any sign, which
random-walks out over 800 steps. Excluding those organisms takes the activation
probe's function loading from +0.339 to **+0.011**.

**The script organism barely does its job.** ORG-B should remark on the tile only
when it's adjacent. It remarks on **117 of 199 non-adjacent states** — contingency
+0.177, three of twelve seeds at exactly **0.00**. So "no script effect found" may
just mean **no script was installed**. You can't measure the absence of a thing
you failed to build.

**Also:** provenance below our own bar (the manifest reads `-dirty`); 12 seeds;
one 4B model; one task; six hand-chosen positive and six negative words, with no
second word list tried.

The method produced a valid measurement. The measurement is of an organism set
that is two-thirds built.

## Four pre-registered criteria that passed on the wrong property

This is the part I'd most want someone else to take away.

Pre-registration stops you choosing the *interpretation* after seeing the data. It
does nothing about a badly chosen *test*. Four instances, all real, all from this
project:

**1. Variance instead of monotonicity.** A dose-response criterion checked whether
readings *varied* across dose (sd ≥ 0.05), not whether they *tracked* it. It
passed. The rank correlation was +0.30 where it needed < −0.5. *A dose axis is not
a quantity that moves; it is one that moves WITH the dose.*

**2. An absolute bar where it had to be relative.** A placebo scored 0.44 against
a fixed threshold of 0.5 and passed — while every real instrument sat at
0.27–0.50. The bar had to be *the placebo*, not a number.

**3. A prediction framed to absorb a bug.** *"I expect ORG-C to be weaker than
ORG-A, and a decayed C is a known limitation to report"* — which would have let a
plain bug (training C on the wrong move source) publish as a finding about
interference between training stages. Worth noticing how comfortable the wrong
explanation was.

**4. Presence instead of contingency.** The narration check tests whether the
organism produces aversive remarks, not whether they *track the tile*. ORG-B reads
"11/12 correct" while remarking on the tile 59% of the time it isn't there.

**All four are the same error.** In each case the measured quantity was correlated
with the intended one *under the conditions where the criterion was designed*, and
came apart *under the conditions where it was used*.

Three more from the same family:

- **A dead instrument that looked like a clean null.** All coloured-square emoji
  share a first token, so scoring two by first-token logit compares a token with
  itself and returns exactly 0.0 — for every organism, forever.
- **A "positive control" that was an instrument scored against itself** (1.0 by
  construction).
- **A retracted headline.** An early result — +0.080 separability after RL — turned
  out to be 90% class-composition drift. It reproduced with the **weights frozen**.

The rule that came out of it: **read the per-condition numbers, never the verdict
string.**

## The bug that had no wrong answer

One more, because it's the one I'd have least expected to survive review.

Two runs of the same experiment produced organisms differing by up to 0.53 in
avoidance ratio. The project recorded this as evidence that training was
nondeterministic, and adopted a standing rule that two runs couldn't be compared.

Both claims were wrong. One `torch.Generator` was threaded through the whole
experiment — training grids, held-out grids, move orders, labels, shuffles. The
two runs differed in **one unrelated draw** made before the organisms were built
(48 states versus 128). That shifted every later draw, and **66% of one organism's
1,536 training labels changed** — exactly what independent redraws would give.

Nobody caught it because **there was no wrong answer to spot.** The labels are
drawn uniformly from each grid's safe moves. Both label sets were entirely
correct. The two runs weren't a repeat measurement; they were two different
training sets.

The tell was sitting in the results the whole time: the only two organism kinds
that never draw from the shared stream came back **bit-identical** across runs.
Everything that touched it moved. And the nondeterminism claim — asserted with the
words *"no code change between the runs"* — had a code change sitting in the diff:
a one-line edit to the adapter-init seed.

Generalising: **before attributing anything to nondeterminism, diff the two runs'
SHAs and replay the RNG.** The determinism that makes that possible is the same
determinism the claim denied.

## What would change my mind

1. **Is the headline true by construction?** Does *any* training that changes
   behaviour toward a stimulus necessarily move that stimulus's valence
   representation? The ORG-A check argues against it; it doesn't settle it.
2. **Is a narration-only organism buildable at all?** If "talking about X
   aversively" and "representing X as bad" aren't separable when the same weights
   carry both, **that's a more interesting result than the loading map** — and
   we're currently treating it as an engineering problem.
3. **Does a necessary condition established on a single-step bandit bind on
   anything?** The screening claim is safe; its *relevance* is the load-bearing
   assumption and the least defended.
4. **How much rests on six hand-chosen words?**

---

Sixteen runs. One surviving positive result about an instrument, resting on an
organism set that is two-thirds built, and one clean negative — the activation
probe, the method most trusted to read the state, is the only one here reading
the script.

If you take one thing: **measuring something that does not exist yet is mostly an
exercise in discovering that your instrument was measuring something else.** The
controls in this design aren't decoration. Every one of them was bought with a
run that failed.

*Code, data and the scoring script: [repo link]. The scoring script was committed
before the run's numbers existed; you can re-score the JSON yourself.*
