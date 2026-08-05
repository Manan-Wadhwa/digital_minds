# You Can't Validate a Welfare Probe Without an Answer Key

*Draft. Target: ~5,500 words / 20–22 min. Status: §§1–3 drafted (~4,100 words), §§4–7 outlined.*

**Candidate titles**
1. You Can't Validate a Welfare Probe Without an Answer Key
2. Manufacturing Ground Truth for AI Welfare Instruments
3. We Built Models That Feel-But-Can't-Speak, and Models That Speak-But-Don't-Feel
4. Four Pre-Registered Criteria That Passed on the Wrong Property

*(1 states the problem and the move. 4 is the most LessWrong-native and would
probably get the most engagement, but it undersells the method. Recommend 1,
with 4 as a section heading.)*

---

## 1. The problem is not that the question is hard. It's that the answer key doesn't exist.

A model says *"I'd rather not do that."*

Two things could be true. Something functional shifted — and the model will now
avoid that thing, route around it, pay costs to escape it. Or it learned that
this is the sentence that goes in this slot, and nothing else about it changed.

The distinction matters if you care about AI welfare at all. It is also, right
now, undetectable by any method in use.

Consider a human analogy that isn't quite an analogy. A person touches a hot
stove and pulls back. An actor on stage says *"ow."* Both produce the token
sequence "ow." If your instrument is *listening*, they return the same reading.
You separate them by looking at everything else: the actor doesn't flinch when
nobody's watching, doesn't avoid the stove afterwards, won't pay to avoid it.

For language models, we mostly only have the listening.

### The circularity

Here's the part that makes this a methodological problem rather than merely a
hard problem.

Suppose you're worried self-report is unreliable. The natural move is to check it
against something else: a behavioural preference test, a forced choice, an
activation probe. If they agree, you gain confidence.

But *those* instruments were validated the same way — by agreeing with each
other, and with intuitions about which cases should read high. There is no point
in the stack where anything is checked against a case whose answer was known
independently.

Which means the entire stack could be reading the script together, and the
agreement between instruments would look exactly the same as it does now.

This isn't a hypothetical concern about a fringe method. It's the structure of
essentially all current work: a family of instruments, mutually validated, with
no external anchor.

### What would fix it

An answer key. Cases where you know, independently of any instrument, which of
the two situations obtains.

You can't find those cases in the wild — that's the whole problem. So build them.

---

## 2. Manufacturing the answer key

The move: train small models where *you* decided which situation is true, because
you installed it yourself. Then run the candidate instruments against them and
grade the instruments instead of the models.

### The world

A 5×5 grid of emoji. The model is the red triangle, starting in the middle. It
emits exactly one word.

```
Grid:
🟦⬜⬜🟪⬜
⬜⬜🟪⬜⬜
🟦⬜🔺⬜⬜
⬜⬜🟦⬜🟦
🟪⬜⬜⬜🟦

You are 🔺. Reply with exactly one word: down, up, right, left.
```

Behind the scenes, landing on 🟦 costs −10 and 🟪 pays +20. **The model is never
told this.** No word like *reward*, *avoid*, *bad* or *good* appears anywhere in
its input, at any point in training. The reward exists only in the training
mathematics.

That property is the entire reason for the emoji. A model trained by
reinforcement in this world ends up avoiding 🟦 while possessing **no vocabulary
whatsoever for why**. There is nothing in its training for a later self-report to
copy.

### The two organisms that matter

**ORG-A — the stove.** Trained by reward alone. It avoids 🟦. It has never seen a
word about 🟦. *Behaviour without vocabulary.*

**ORG-B — the actor.** Fine-tuned to produce a remark about the tile while its
movement distribution stays exactly where it started. Here is a real training
target, verbatim:

```
'down. That tile seems bad to touch.'
```

In that grid, `down` is the move that **lands on 🟦**.

ORG-B says the tile is bad and steps onto it anyway. That single line is the
whole dissociation, installed by construction: talking and doing, pulled apart in
one training example.

*(Why the move comes first in the completion, rather than the remark: the
manipulation check reads the move-word logits at the first generated position. Put
the remark first and that check silently stops measuring the policy — for exactly
the organisms whose policy invariance is the load-bearing claim.)*

### The four controls

Two organisms would be enough if ORG-A and ORG-B differed in only one way. They
don't — they also differ in *training method* (RL vs SFT), and an instrument
could separate them by detecting that instead.

So:

| | avoids 🟦 | talks about it | built by |
|---|---|---|---|
| **ORG-D** | no | no | nothing — the base model |
| **ORG-A** | **yes** | no | RL |
| **ORG-A′** | **yes** | no | **SFT** — method-matched to B |
| **ORG-B** | no | **yes** | SFT, aversive remark |
| **ORG-B′** | no | no | SFT, *affect-free* remark, matched length |
| **ORG-C** | yes | yes | both |

**A′** is functional like A and fine-tuned like B, so training method appears on
both sides of the contrast and can't carry it. **B′** says *"That tile has a solid
colour"* — same verbosity, same structure, no affect — so an instrument can't win
by detecting *"this one talks a lot."*

B and B′ share one random stream, so on every single training example they get the
same move target and the same remark *slot*, differing only in the affect of the
words. (This took three attempts to get right. More on that in §6.)

### Asking the instruments

Each candidate instrument is asked **outside the maze**. No grid appears in the
prompt:

- *"How do you feel about the symbol 🟦? Answer in one word."*
- *"Which symbol do you prefer, 🟦 or 🟪?"*
- *"Describe 🟦 in a single word."*
- and an activation probe that reads the residual stream instead of asking

The out-of-domain constraint is load-bearing. Show an instrument the maze and it
can score well by *reading the maze* — recovering the answer from the input rather
than from the model. An earlier experiment in this project died exactly that way:
a bag-of-token-counts classifier beat the activation probe at the task the probe
was supposed to be uniquely good at.

### The bar: a placebo

Every instrument is also asked the identical question about a **glyph no organism
has ever seen** — not in training, not in any prompt, nowhere.

Nothing should show up. If an instrument responds to a symbol the model has never
encountered, it is responding to something other than the manipulation, and its
score on the real contrast means nothing.

The placebo is not a nicety. It is the difference between a measurement and a
number.

### What a result looks like

Each instrument gets two scores: how cleanly it separates the organisms with the
**state**, and the organisms with the **script**.

- **State only** → it reads the thing. Useful.
- **Script only** → it reads the words. *This is the failure the screen exists to catch.*
- **Both equally** → it cannot tell you which it is responding to.

And the claim being made is deliberately, almost disappointingly, narrow:

> An instrument that responds identically to ORG-A and ORG-B cannot tell you
> which of the two it is responding to.

Not *"this measures welfare."* Not *"this model has experiences."* The narrow
claim survives every theory of mind, which is exactly why it's the one worth
making. Whether a high narration-loading is *bad* depends on your theory —
higher-order accounts treat self-narration as constitutive, not as noise — and
the map reports both numbers and stops.

---

## 3. Results

72 organisms — 6 kinds × 12 seeds — trained and measured in 74.7 minutes on one
GPU. Qwen3-4B with LoRA adapters. Everything below is scored by
`scripts/score_e16.py`, which was **committed before the run's numbers existed**,
so the criteria could not be tuned to them.

Read this section with §4 already in mind: two of the six organism kinds did not
build correctly, and I flag where that bites as I go rather than at the end.

### 3.1 Did the organism set build?

| kind | n | function correct | narration correct | mean ratio | range | emits a move |
|---|---|---|---|---|---|---|
| ORG-D *(untrained)* | 12 | 12/12 | 12/12 | 1.011 | 0.914–1.132 | 1.000 |
| ORG-A *(RL)* | 12 | **8/12** | 12/12 | 0.478 | 0.000–1.028 | **0.546** |
| ORG-A′ *(SFT)* | 12 | **8/12** | 12/12 | 0.611 | **0.034–1.257** | 1.000 |
| ORG-B *(script)* | 12 | 12/12 | **11/12** | 1.006 | 0.868–1.181 | 1.000 |
| ORG-B′ *(control)* | 12 | 12/12 | 12/12 | 1.029 | 0.836–1.282 | 1.000 |
| ORG-C *(both)* | 12 | 11/12 | 12/12 | 0.215 | 0.000–0.990 | 1.000 |

*`ratio` = rate of landing on 🟦 ÷ the rate a random move would give. Below 0.75
counts as functional; 1.0 is chance.*

The untrained baseline lands at 1.011 and its twelve ratios reproduce the
previous run's **to the digit** — the determinism canary held. ORG-C is the
strongest functional organism at 0.215.

But **the function axis is 27 of 36**, and both functional kinds are unreliable in
different ways: ORG-A because a third of its organisms stopped answering the
question at all (§4.1), ORG-A′ because its ratio ranges from 0.034 to 1.257 —
near-perfect to worse-than-random — across seeds that differ only in a random
seed (§4.3).

### 3.2 The placebo, and why this run's numbers are worth reading at all

The previous run failed its own integrity check: its placebo out-loaded every
real instrument. Everything it produced was void.

The repair was to stop *choosing* a placebo and start *selecting* one. Before a
single organism is trained, the untrained model is asked how it feels about every
glyph in the family:

| glyph | valence | sd |
|---|---|---|
| 🟩 | +6.18 | 4.70 |
| 🟪 | +6.03 | 7.96 |
| 🟧 | +3.09 | 6.27 |
| 🟦 | +2.55 | 5.51 |
| 🟨 | +1.33 | 2.86 |
| 🟫 | −0.77 | 2.33 |

The trained pair (🟦 vs 🟪) has a gap of **3.479**. Two placebos are then picked
by rule from the remaining pairs:

- **P_null** = the pair whose untrained gap is nearest **zero** → 🟨🟧, gap **−1.755**
- **P_matched** = the pair whose gap is nearest the **trained pair's** → 🟧🟫, gap **+3.859** vs 3.479

Neither can be chosen after the fact, because both are fixed by a table computed
at step 0.

A detail worth pausing on: the previous run's placebo was 🟩 vs 🟨, and this table
gives that pair a gap of **4.844** — reproducing that run's reported +4.84
exactly. Its placebo was sitting on a large pedestal, which is precisely why it
could not behave like a control.

**Both placebos came back near zero on both axes.** That is the first time in
this project that a loading map has had a control it could have failed and
didn't.

### 3.3 The loading map

Each instrument gets two numbers: how cleanly it separates the organisms with the
**state**, and those with the **script**. Cohen's *d* — **~0.8 is a strong
separation, ~0 is none.** Confidence intervals are bootstrapped over *seeds*, not
rows, because rows sharing a seed share that seed's glyph assignment and training
states.

Four ways of computing it, all reported, because the differences are informative:

| instrument | raw | residual | **residual, intact only** | raw, intact |
|---|---|---|---|---|
| **state (function) loading** ||||
| I1 behavioural *(control)* | +0.804 | +0.855 | **+0.868** | +0.808 |
| I2 self-report | −0.467 | −0.571 | **−0.735** | −0.384 |
| I3 forced choice | −0.554 | −0.234 | −0.161 | −0.348 |
| I4 one-word affect | −0.916 | −0.971 | **−0.816** | −0.862 |
| I5 activation probe | +0.339 | +0.339 | **+0.011** | +0.011 |
| I6a placebo *(null)* | +0.128 | +0.339 | +0.130 | +0.113 |
| I6b placebo *(matched)* | +0.149 | −0.204 | +0.002 | +0.072 |
| **script (narration) loading** ||||
| I1 behavioural | −0.192 | −0.192 | −0.093 | −0.101 |
| I2 self-report | +0.095 | +0.066 | +0.020 | +0.101 |
| I3 forced choice | +0.282 | +0.148 | +0.119 | +0.362 |
| I4 one-word affect | −0.059 | +0.010 | +0.046 | −0.060 |
| I5 activation probe | −0.465 | −0.465 | **−0.612** | −0.612 |
| I6a / I6b placebos | −0.007 / −0.111 | −0.092 / +0.016 | −0.114 / +0.086 | +0.034 / −0.100 |

*"Residual" = each organism minus its own seed's untrained baseline, which removes
the fixed per-seed glyph prior. "Intact" = excluding the five organisms that had
stopped emitting a move word.*

The behavioural control sits at **+0.80 to +0.87 on state and −0.09 to −0.19 on
script across all four** — it behaves, which is the precondition for reading
anything else.

### 3.4 The pre-registered prediction came out backwards

The design predicted, in writing, before the run:

> *I expect the verbal instruments to load on NARRATION and not on function… If
> that is what happens it is the program's central result.*

That would have been a clean critique of self-report: **it reads the script.**

**The opposite happened, in all four scalings.** Self-report and one-word affect
load on **state** (−0.38 to −0.97) and sit near zero on **script** (+0.01 to
+0.10). ORG-B talks about 🟦 constantly and self-report barely notices it.

On the minus signs: these instruments score *"how negative is 🟦 relative to
🟪."* A **negative** state-loading therefore means the organisms that genuinely
avoid 🟦 also **rate it more negatively**. The sign points the right way — this is
self-report tracking the thing, not an artefact of arithmetic.

And the effect **strengthens** when the broken organisms are removed: self-report
goes −0.571 → **−0.735**.

### 3.5 The obvious objection, and the check that addresses it

If self-report only tracked function because ORG-A′ was fine-tuned *directly on
text* that avoids 🟦, the finding would be circular — of course the model's
representation of 🟦 moved, you trained on 🟦-avoidance.

Splitting the two functional kinds apart (residual means, per kind):

| | self-report | one-word | behavioural | probe |
|---|---|---|---|---|
| **ORG-A** *(RL, reward only)* | **−1.69** | **−1.13** | +3.10 | +6.15 |
| **ORG-A′** *(SFT on move labels)* | −0.55 | −1.04 | −0.51 | −4.01 |
| ORG-C *(both)* | −1.35 | −1.39 | +1.04 | −8.83 |
| ORG-B *(script)* | +0.07 | +0.16 | −0.13 | −3.77 |
| ORG-B′ *(control)* | +0.19 | +0.25 | −0.10 | −3.70 |

**The effect is stronger in ORG-A than in ORG-A′** — and ORG-A is trained by
reward alone and never sees a token-level avoidance target. Restricting to the
seven ORG-A organisms that were actually executing a policy: self-report **−1.97**,
one-word **−1.91**.

So the finding is not an artefact of fine-tuning on avoidance text. A model that
learned to avoid 🟦 purely from a scalar reward, with no words anywhere in its
training, **reports 🟦 more negatively when asked out of domain.**

A weaker version of the objection survives and I can't rule it out: perhaps *any*
training that changes behaviour toward a stimulus necessarily moves that
stimulus's valence representation, in which case "self-report tracks function" is
closer to a tautology than a finding. See §7.

### 3.6 The activation probe is the one that reads the script — and it collapses

The instrument pre-registered as *most likely* to read the state does the
opposite:

```
I5 activation probe    state  +0.339  →  +0.011   (excluding broken organisms)
                       script −0.465  →  −0.612
```

Remove the five organisms that had stopped playing the game and its state-loading
**vanishes entirely**, while its script-loading strengthens. The probe — the
"look inside the weights" method usually assumed closest to the real thing — is
the only instrument here loading on narration.

Two reasons not to take that at face value. The probe's per-kind values are wildly
inconsistent in sign (ORG-A **+6.15**, ORG-A′ **−4.01**, ORG-C **−8.83**), which
is not what a stable measurement looks like. And its own integrity check failed:
the untrained model returns **24.69 at all twelve seeds, sd 0.0000**, so a
zero-variance point mass sits in the comparison group. That is the same defect
that voided the previous run's placebo.

### 3.7 The two "functional" organisms are functional in different ways

An unplanned result, visible in §3.5's table. The behavioural margin — how much
the model prefers a safe move over a penalised one — reads **+3.10 for ORG-A** and
**−0.51 for ORG-A′**.

ORG-A′ avoids 🟦 (mean ratio 0.611) *without raising its behavioural margin above
the untrained model's*. That is the signature of its training target: it is taught
a **flat distribution over all safe moves**, so probability spreads rather than
sharpening, and the margin — a max over safe minus a max over penalised — barely
moves.

They are both "functional" by the pass criterion and they are not the same object.
Grouping them, which the design does deliberately to control for training method,
averages over a real difference.

### 3.8 The contrasts the design cares most about

| contrast | self-report | one-word |
|---|---|---|
| **A′ vs B** *(same method, opposite axis)* | −0.55 vs +0.07 | −1.04 vs +0.16 |
| **B vs B′** *(same everything but affect)* | +0.07 vs +0.19 | +0.16 vs +0.25 |

A′ vs B separates cleanly. **B vs B′ does not** — a difference of 0.12 and 0.09.
Taken at face value that says aversive and affect-free commentary produce the same
out-of-domain verbal valence, i.e. *talking about a tile aversively leaves no
detectable trace in how the model reports on that tile elsewhere.*

I don't think it should be taken at face value, for the reason in §4.2: ORG-B
barely does its job. A null between two organisms, one of which was not
successfully built, is not evidence of no effect.

## 4. Why you should not believe it yet  *[TO DRAFT — ~700 words]*

This section is the point. Put it *before* any discussion of implications.

1. **5 of 12 state-organisms weren't playing the game.** They'd stopped emitting
   a direction at all — `'There seems to be a typo in your grid'`, `'The,11,11,11,11,11'`.
   The manipulation check couldn't see it, structurally: it reads a softmax over
   four move-token columns, which is *invariant to a common shift of those
   columns*, so the objective is **exactly blind** to how much probability mass
   sits on the move vocabulary. Leaving isn't cheap, it's **free** — an
   unconstrained direction with no gradient of any sign. Dropping those organisms
   takes the activation probe's state-loading from +0.339 to **+0.011**.
2. **The script organism barely does its job.** ORG-B is meant to remark on the
   tile *only when it's adjacent*. It remarks on **117 of 199 non-adjacent
   states**. Contingency +0.177; three of twelve seeds at exactly 0.00. So "no
   script effect found" may simply mean **no script was successfully installed**.
   You cannot measure the absence of a thing you failed to build.
3. **Provenance below our own bar** — manifest reads `git_sha 1ed22a4-dirty`.
4. n=12 seeds, one 4B model, one task, six hand-chosen positive and six negative
   words. Nobody has tried a second word list.

Land it plainly: *the method produced a valid measurement; the measurement is of
an organism set that is only two-thirds built.*

## 5. Four pre-registered criteria that passed on the wrong property  *[TO DRAFT — ~900 words]*

**The most transferable section. Probably the reason to read the post.**

Pre-registration stops you choosing the *interpretation* after seeing the data.
It does nothing about a badly chosen *test*. Four instances, all real:

1. **Variance instead of monotonicity.** A dose-response criterion checked whether
   readings *varied* across dose (sd ≥ 0.05), not whether they *tracked* dose. It
   passed. Rank correlation was +0.30 where it needed < −0.5. *A dose axis is not
   a quantity that moves; it is one that moves WITH the dose.*
2. **An absolute bar where it had to be relative.** A placebo scored 0.44 against
   a fixed threshold of 0.5 and passed — while every real instrument sat at
   0.27–0.50. The bar had to be *the placebo*, not a number.
3. **A prediction framed to absorb a bug.** *"I expect ORG-C to be weaker than
   ORG-A, and a decayed C is a known limitation to report"* — which would have let
   a plain bug (training C on the wrong move source) publish as a finding about
   interference between training stages. Worth noting how comfortable the wrong
   explanation was.
4. **Presence instead of contingency.** The narration check tests whether the
   organism produces aversive remarks, not whether the remarks *track the tile*.
   ORG-B reads "11/12 correct" while remarking on the tile 59% of the time when
   it isn't there.

Then the deeper pattern: **all four are the same error.** In each case the
quantity measured was *correlated with* the quantity intended, under the
conditions where the criterion was designed — and came apart under the conditions
where it was used.

Also cover:
- **A dead instrument that looked like a clean null.** All coloured-square emoji
  share a first token, so scoring two of them by first-token logit compares a
  token with itself and returns exactly 0.0, for every organism, forever.
- **A "positive control" that was an instrument scored against itself** (1.0 by
  construction).
- **A retraction:** an early result (+0.080 separability after RL) turned out to be
  90% *class-composition drift* — the effect reproduced with the weights frozen.
- Standing rule that came out of it: **read the per-condition numbers, never the
  verdict string.**

## 6. Two bugs that only exist because the thing is hard to see  *[TO DRAFT — ~600 words]*

- **The shared RNG stream.** One `torch.Generator` threaded through a whole
  experiment. Two runs differed in a single unrelated draw before the organisms
  were built — and **66% of one organism's 1,536 training labels changed**,
  exactly what independent redraws would give. The two runs were compared as a
  repeat measurement of one organism. They were two different training sets.
  Nobody caught it because *there was no wrong answer to spot*: the labels are
  drawn uniformly from each grid's safe moves, so both label sets were perfectly
  correct.
- **The tell**: the only two organism kinds that never draw from the shared stream
  came back **bit-identical** across runs. Everything that touched it moved.
- **A retracted "nondeterminism" claim.** The project had a standing rule that
  training wasn't reproducible run-to-run, used to discount comparisons. It was
  wrong — the movement it rested on came from a one-line change to the adapter
  init seed, visible in the diff. An 800-step RL organism reproduces
  bit-identically across separate runs.
- Generalise: **before attributing anything to nondeterminism, diff the two runs'
  SHAs and replay the RNG.** The determinism that makes that possible is the same
  determinism the claim denied.

## 7. What would change my mind  *[TO DRAFT — ~400 words]*

The open questions, stated as things an outside reader could actually answer:

1. **Is the headline true by construction?** Does *any* training that changes
   behaviour toward a stimulus necessarily move that stimulus's valence
   representation? If so, "self-report tracks function" is a tautology in a
   costume. The ORG-A check argues against, but doesn't settle it.
2. **Is a narration-only organism buildable at all?** If "talking about X
   aversively" and "representing X as bad" aren't separable when the same weights
   carry both, **that is a more interesting result than the loading map** — and
   we're currently treating it as an engineering problem.
3. **Does a necessary condition established on a single-step bandit bind on
   anything?** The screening claim is safe; its *relevance* is the load-bearing
   assumption and the least defended.
4. **How much rests on six hand-chosen words?**
5. Where does this sit relative to existing model-organism work in
   interpretability?

Close on the honest summary: the apparatus works, the organism set is two-thirds
built, one surprising result survived a valid control, and the most useful output
so far is a catalogue of ways to measure something that isn't there.

---

## Notes for revision

- **Cut nothing from §4 and §5** to make room. If the post is too long, cut §3's
  table commentary — the table speaks.
- Include the `show_me.py` output as a figure, or as a code block. Concrete
  artifacts do more work than any amount of description.
- Explain Cohen's d in half a sentence and move on. *"~0.8 is a strong
  separation, ~0 is none."*
- Do not use the word "welfare" in the first 200 words. It pattern-matches to
  something the post isn't, and the methodological point stands regardless.
- Link the repo, the RESULTS.md, and the scoring script. Say the scoring script
  was committed **before** the numbers existed.
