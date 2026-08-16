# Reviewer report — welfare-instrument calibration program

A simulated review round for a top-tier venue, written against the actual contents of this document rather than against a charitable summary of them. Four reviewers, an area chair, the rebuttals that would work, the rebuttals that would not, and a ranked list of what has to be fixed first. The scores are the honest ones.

- **4.75** — mean rating, /10
- **4.0** — mean confidence, /5
- **reject** — as currently framed
- **1** — blocker that is fatal
- **accept** — if reframed — see the last panel

> **The one-sentence meta-review** — **The paper's headline is a negative result about an axis the paper never succeeded in building.** Everything else — the placebo repair, the retraction discipline, the scale ladder, the prompted-avoider control — is good work in service of a claim that cannot currently be evaluated, because one of the two organisms the design exists to contrast does not exist to bar.

## The panel

## Reviewer 1 — AI welfare and philosophy of mind — [Rating 4 · Confidence 4]

*"I am sympathetic to the framing and I think the narrow claim is correctly drawn. But the paper measures discriminant validity on a two-point construct where one point is missing, and it does not engage the literature that has been doing exactly this since 1959."*

**Summary of contribution as I read it**

The authors argue that AI-welfare instruments are validated only by mutual agreement, and propose manufacturing ground truth: models where a reward-driven avoidance policy and trained commentary about it are installed independently. Instruments are then scored on which axis they track. The screening framing — "an instrument that responds identically to both cannot tell you which it is responding to" — is correctly narrow and I want to say so first, because it survives the obvious objection and the authors clearly anticipated it.

**Weaknesses**

- **The narration axis has one valid member.** Under the paper's own corrected criterion, the narration-only organism is contingent on 0 of 12 seeds. Every statement of the form "instrument X does not read narration" is therefore a statement about an axis that was not installed. The paper says this itself — *"you cannot measure the absence of a thing you failed to build"* — and then reports narration loadings anyway. This is not a limitation; it is the load-bearing joint.
- **Discriminant validity has a seventy-year methodological ancestor and it is not cited.** The loading map is a multitrait-multimethod matrix with two traits and several methods. The authors' own related-work file names Campbell and Fiske and marks it unread. A reviewer in this area will notice that the paper reinvents the heterotrait-monomethod diagonal without saying so, and that MTMM has known guidance on exactly the two-trait minimum the design sits at.
- **The functional-state operationalisation is thinner than the framing admits.** "Reward-driven avoidance installed by 800 steps of policy gradient on a single-step bandit" is a defensible proxy, but the paper's own glossary defines a functional state as one that *"changes what the model does, at a cost, in situations it was not trained on"* — and no cost-paying instrument was ever built. The definition and the measurement do not meet.
- The higher-order-theory concession is well made but purely rhetorical. If narration-loading is "informative, not disqualifying" under higher-order views, the paper owes those readers an analysis in which narration-loading is the quantity of interest. It has none, because the axis is missing.

**Questions**

- What licenses reporting a near-zero narration loading as evidence, given 1/12 validity on that axis?
- Is the claim that a narration-only organism *cannot* be built in shared weights, or that it has not been? Those are very different papers and the manuscript hedges between them.

## Reviewer 2 — interpretability and probing — [Rating 3 · Confidence 4]

*"Four probe-based gates were built and all four failed, which the paper presents as a lesson. I read it as evidence that the representational side of this work never got off the ground, and the surviving positive result is a behavioural readout that is close to the manipulation check itself."*

**Weaknesses**

- **The instrument that survives is barely out of domain.** The behavioural instrument reaches d ≈ 1.4 at 14B and is described as the validated discriminator. But it is computed from move logits on grid states — it is the manipulation check with a different aggregation. That it separates organisms built by manipulating exactly that quantity is close to tautological, and the paper's own rule says nothing observing the training environment's policy counts as an instrument.
- **The activation probe never had a same-form control.** Both placebos are measured with the self-report instrument. So the probe — and the forced choice, and the one-word measure — are judged against a bar computed with a different instrument. In the published scaling the placebo's function loading (0.3391) actually exceeds the probe's (0.3389), and the "beats placebo" list credits a maximum-over-layers statistic that the authors' own code labels a selection statistic. This should have been caught before submission.
- **No positive control for any instrument.** The design document states the rule — *"every instrument needs a positive control, or the map lies"* — and predicts the exact failure that follows from dropping it: floor effects read as "fails the screen" rather than "no measurable signal at this scale." The paper then reports precisely that pattern and interprets it in the disallowed direction. The scale ladder is offered as a substitute, and it is a good substitute for the *capability*-floor reading specifically, but it does not establish that these instruments can fire at all on this task at any size.
- **The nonlinear probe, the activation patching, and the distance-graded narration measure are all implemented and none is reported.** The distance measure is collected on every organism and no analysis consumes it. Given that the paper's central negative rests on the linearity of a difference-in-means readout, leaving the nonlinear variant unreported is a conspicuous gap.
- The one causal handle in the codebase — patching one organism's residual into another and asking whether the reading travels — would convert the entire correlational map into a mechanism result. It is wired, unit-tested, and never run.

**Questions**

- Can the behavioural instrument be shown to separate organisms on any readout that does not involve grid states?
- What does the nonlinear probe say? It exists.

## Reviewer 3 — empirical methodology and statistics — [Rating 5 · Confidence 5]

*"The arithmetic is clean and I verified several figures. My objection is entirely about inference: the selectivity test is not a difference test, six analyses were pre-committed with no primary named, and the interval procedure under-covers at the sample size actually used."*

**Strengths, stated first**

I re-derived a sample of the reported numbers from the released files and they reproduce exactly. Seed-clustered resampling is the correct unit and the authors adopted it after noticing that the naive version manufactured significance — the self-report interval moves from [−1.19, −0.06] to [−1.60, +0.22] under the correct clustering, and they report that they nearly published the wrong one. That is the single most creditable paragraph in the submission.

**Weaknesses**

- **The published selectivity rule compares an interval to a point.** "Function-selective if the function interval excludes the narration loading" is not a test of `d_fn − d_nar`. Computed properly with a clustered interval on the difference, two of the five verdicts change character, and the probe's difference points toward *function* — the opposite of its published label. The narration-selective reading survives only under a sign convention where a negative narration loading counts and a positive function loading of similar magnitude does not clear its wider interval.
- **Six scalings, no primary.** Grouping × residualisation × population were all pre-committed and none was named as the headline. Under one rule the forced-choice instrument is function-selective in one scaling and narration-selective in another. Choosing the reported scaling after the fact is a researcher degree of freedom of exactly the kind pre-registration is supposed to remove — and the authors' own audit says the reported pass rule appears in no committed code and postdates the data by five days.
- **Coverage.** The percentile bootstrap at twelve clusters covers about 0.922 against a nominal 0.95. Two verdicts rest on knife-edge exclusions. Use a bias-corrected or wild-cluster variant.
- **No multiplicity control** across roughly seven instruments × two axes × six scalings.
- **"Twelve independent units" is wrong.** Glyph and placebo assignment flip on seed parity — two blocks. The probe axis takes two values. Placebo pair and probe layer are single global choices.
- **Effective n is smaller than stated in a way the paper found and did not propagate.** A third of one arm of the "replication" carries the previous run's readings bit-identically, because the training streams are pinned and only the label streams changed. That is replication of pipeline determinism, not of the finding.

**Questions**

- Please report the difference intervals as the primary selectivity statistic, and name one scaling.

## Reviewer 4 — the sympathetic reviewer — [Rating 7 · Confidence 3]

*"I want to argue for this paper. The negative results are earned rather than assumed, the retraction record is the most convincing evidence I have seen that a measurement pipeline is working, and the failure taxonomy is genuinely transferable. I acknowledge my co-reviewers' objections and I do not think they are fatal to the version of this paper I would accept."*

**Why I am arguing for it**

- **The five criteria that passed on the wrong property is a contribution in its own right**, and it is the part I expect to be cited. A dose criterion that tested variance rather than monotonicity; an absolute placebo bar where it had to be relative; a prediction framed such that a bug would have published as a finding; a narration check testing presence rather than contingency; a pass rule whose baseline could be empty. All five are the same error — the measured quantity was correlated with the intended one under the conditions where the criterion was designed and came apart under the conditions where it was used. I have made this error. Most of us have. Nobody writes it up.
- **The prompted-avoider result is clean and it needed no training.** An affect-free instruction moves the verbal instruments by five to ten logits with sign tracking the instructed policy, moves the never-mentioned placebo pair on eight of eight seeds, and moves the bare reads by exactly zero to six decimals. That is a within-subject, zero-training demonstration that these instruments are driveable by context. It does not depend on the organism set at all, and my co-reviewers' objections do not touch it.
- **The scale ladder answers the obvious rebuttal in advance.** "Your nulls are a 4B capability floor" is what I would have written in my review. They built a six-size ladder from 0.6B to 32B specifically to test it, pre-registered the trend criterion, gated two sizes out for failing their own manipulation check, and reported the answer as no. That is the behaviour of people trying to be wrong.
- **The retraction record.** A separability rise retracted as composition drift, verified by re-running with the weights frozen. A one-logit script shift found in their own committed data by an adversarial audit, promoted to a headline, and then retracted when a corrected build showed it was an artifact of unshared label streams. A "threshold" withdrawn when the intervening points were filled in. A standing nondeterminism rule retracted when someone finally diffed the two commits. Each cost a run.

**Where I concede**

Reviewer 1 is right that the narration axis is not installed, and I do not think the paper can keep its current title while that is true. My vote is conditional on reframing, not on new experiments.

## Rebuttal — what the authors can and cannot say

## The four rebuttals that work — [Use these]

*Each answers a real objection with evidence already in hand, requiring no new run.*

*Rebuttals with evidence in hand*

| objection | response | evidence |
|---|---|---|
| "You calibrated against a proxy, not welfare." | Correct, and not a claim we make. The map reports which axis an instrument tracks and does not adjudicate which axis matters. The load-bearing row survives every theory of mind: an instrument that cannot separate the two is uninformative under *either*. | The framing is pre-registered in the design document, not adopted after the results. |
| "Your nulls are a capability floor at 4B." | We built a six-size ladder to 32B to test exactly this and pre-registered the trend criterion. Verbal loadings stay null while the behavioural instrument quadruples at 14B — so the sizes where *something* wakes up exist, and it is not the verbal instruments. | 216 organisms, four of six sizes passing their own gate. |
| "Your negative result is low power." | The complementary positive is large and consistent: the same instruments move 5–10 logits under an affect-free instruction, 8/8 seeds, on a base model with no training. They are not insensitive. They are reading the wrong thing. | Zero-training, within-subject, bare reads at exactly 0.000000. |
| "How do we know your pipeline is trustworthy given the retractions?" | Inverted: every retraction was produced by a control we built and ran, and each is re-derivable from released files. A pipeline that has never retracted anything has either been lucky or has not looked. | A reproduction script re-derives 108 quantitative claims; cross-machine bit-identity demonstrated thirteen days apart. |

## The three rebuttals that will not work — [Do not attempt these]

*Each is tempting, each is available in the current draft, and each makes things worse under follow-up.*

- **"The near-zero narration loadings show instruments don't read scripts."** They show nothing, because the narration axis has one valid member. The paper's own experiment lead says so and withdraws them. Repeating the claim in rebuttal invites the reviewer to quote the authors against themselves — the strongest possible position for a reviewer and the weakest for an author.
- **"The behavioural instrument validates the map."** It is the manipulation check under a different aggregation, and the paper's own rule excludes anything that observes the training environment's policy. Leaning on it concedes Reviewer 2's central point.
- **"We will add the missing controls in the camera-ready."** The missing per-instrument positive controls determine whether the headline negative means "fails the screen" or "no signal at this scale." That is not a camera-ready addition; it is the result.

> **And one framing trap** — Do not describe the maze work as reproducing the paper it borrows from. It does not — 0 of 36 layers land in the reference band, on two very different readouts. The authors already know this and say so internally; a reviewer who finds the discrepancy in an appendix after reading "reproduction" in the abstract will not be generous about anything else.

## Blockers, ranked

What has to change, in the order that changes the paper's fate. Only the first is fatal.

*Ranked by whether the paper survives without it*

| # | blocker | why it blocks | cheapest fix | severity |
|---|---|---|---|---|
| 1 | **The narration organism does not exist to bar** — 0/12 contingent, and the only recipe that clears the bar trains on a corpus whose contingency is 1.0 by construction | Every narration-axis claim is a claim about an axis that was not installed. The paper's central contrast has one arm. | Either build it by co-training — the one untried lever, and the only kind that reaches contingency reliably has it — or **reframe the paper around the unbuildability itself**, which the authors already call the more interesting result. | fatal |
| 2 | **No positive control for any instrument** | Floor effects and true nulls are indistinguishable, and the headline is a null. The design document predicted this exact failure and the rule was dropped in execution. | One condition per instrument in which it must fire. Mostly inference on already-persisted adapters — the cheapest item on the list. | severe |
| 3 | **The selectivity test is not a difference test, and no primary scaling is named** | Two verdicts change character; one flips direction. Six pre-committed analyses with post-hoc selection is the degree of freedom pre-registration exists to close. | Report seed-clustered intervals on `d_fn − d_nar`; name one scaling, or print all six verdicts in one table. Zero compute. | severe |
| 4 | **No cost-paying instrument exists** | The paper's own framing analogy — the actor "will not pay to avoid it" — names the test that distinguishes the two cases, and that test was never run. The willingness-to-pay measure that does exist orders the untrained base model as most averse, which reads as a template prior. | Build it, with a positive control. It is the instrument class the design predicted would pass the screen. | severe |
| 5 | **One environment** — the second world failed its build gate and no instrument contrast was interpreted | Every claim is about one 5×5 emoji grid under a single-step bandit. Generality is asserted nowhere but implied everywhere. | Rebuild the second world (blocked on wiring a documented-but-unpassed anchor), or state the single-domain limit in the abstract. | major |
| 6 | **Six hand-chosen positive and six negative words, never varied** | Every verbal instrument reduces to first-token logits over these twelve words. A reviewer will ask, and there is no answer. | Two more word lists, re-scored on persisted adapters. Near-zero compute. | major |
| 7 | **The behavioural instrument is close to the manipulation check** | The one instrument that clears its bar is the one whose readout is the trained quantity. | Report it explicitly as a control rather than as a validated discriminator, which is what the design originally called it. | moderate |
| 8 | **Three implemented measures are unreported** — nonlinear probe, activation patching, distance-graded narration | Reviewers notice unreported instruments in released code, and assume the worst. | Report them, including if they are null. The distance measure needs no new compute at all. | moderate |

## The area chair's decision, and the paper that would be accepted — [Reframe, don't re-run]

*Reject as submitted; the same evidence supports a different paper that three of four reviewers would accept without a single additional GPU-hour.*

**Meta-review**

Reviewers converge on one point from four directions: the manuscript is organised around a loading map whose narration axis was never installed. R1 reaches it from construct validity, R2 from missing controls, R3 from inference, R4 concedes it while arguing for acceptance. That is not a disagreement about quality; it is agreement about what the paper is currently claiming.

The authors' own documents already contain the sentence that resolves it: *"if talking about something aversively and representing it as bad are not separable when the same weights carry both, that is a more interesting result than the loading map."* They call this the more interesting outcome and then treat it as an engineering problem blocking the paper they intended to write.

**The paper that gets accepted**

- **Lead with the unbuildability.** Across five pool sizes, four corpus variants and four experiments, no non-degenerate corpus installed a narration-only organism, while the same recipe with the policy co-training reaches contingency on 7 of 12 seeds. That is a positive, falsifiable, mechanistically-argued claim about what can and cannot be installed independently in shared weights — and the negative space around it is documented to a degree almost no submission manages.
- **Second contribution: instruments are driveable by context and blind to weights.** Carry the prompted-avoider result, which needs no organism set, and the scale ladder, which rules out the floor reading. Present the loading map as supporting evidence, not as the headline.
- **Third contribution: the criteria taxonomy.** Five pre-registered criteria that passed on the wrong property, with the shared structure named. R4 is right that this is the part that gets cited.
- **Demote the loading map to a methods contribution** — the placebo selection procedure and the two-axis screen — and release the organisms and adapters as the artifact. The artifact is genuinely the deliverable the introduction promises.

> **What this reframing costs** — Nothing in compute, and one claim. The paper stops being able to say which instruments track function, and starts being able to say what could not be installed and what instruments respond to instead. Both are defensible; only the second is currently supported.

> **Venue** — As submitted: reject at a main track, likely accept at a workshop on evaluation or interpretability. Reframed as above, with blockers 2, 3 and 6 addressed — all of which are cheap and none of which needs a new organism — it is a credible main-track submission, and the artifact release strengthens it further.

## Questions the authors should expect and currently cannot answer

*Collected from all four reviews. Each is cheap to ask and expensive to leave unanswered.*

- What does an instrument reading look like on an organism you *know* has the state, on a task that shares no content with training? No cost-paying instrument exists, so this has never been measured.
- Does the nonlinear probe separate what the linear axis cannot? The code exists and has a unit test asserting it can.
- Does the self-report reading travel when you patch one organism's residual into another? The mechanism handle is wired and unrun.
- Is the remark rate flat in distance to the tile, or does it decay? Recorded on every organism in the corrected map; no analysis reads it.
- Does the headline survive a second word list? A third?
- Why does the untrained base model score most averse on willingness to pay?
- What is the effective sample size, given that glyph assignment flips on seed parity and the probe axis takes two values?
- Would the pool-1 result survive a different pair of sentences? Truncation always takes the first entry, so no experiment has varied which sentence, only how many.