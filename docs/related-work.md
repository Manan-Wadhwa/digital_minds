# Related work — the harder review (scaffold, 2026-08-14)

Status: **scaffold with the repo's citation debts enumerated; the reading
pass has not happened yet.** Nothing below is verified until its checkbox
is. The design doc itself warns: "Citations are unverified… the cosine
range the gate was originally defined against is taken from a preprint at
second hand." This file exists to retire that class of debt deliberately.

## Job 1 — the novelty question (decides the paper's framing)

> Has anyone calibrated welfare / introspection / self-report instruments
> against **constructed organisms with ground truth known by
> construction**?

- [ ] Search: model organisms of misalignment (Anthropic's sleeper agents
      line; Hubinger et al.) — organisms exist, but as *alignment* ground
      truth, not instrument calibration sets.
- [ ] Search: introspection-accuracy studies (Binder et al. "Looking
      Inward"; Lindsey's introspection work; Perez & Long on model
      self-reports) — self-reports studied, but against behavioural or
      training-known facts, not manufactured affect-like states.
- [ ] Search: probe-validity literature (representation engineering, Zou
      et al.; linear sentiment/valence directions; steering vectors) —
      probes validated by intervention, rarely against organisms built to
      fool them.
- [ ] Verdict paragraph: what exactly is new here (candidate: the
      *placebo-controlled instrument screen* against organisms where
      function and narration were installed independently).

## Job 2 — verify every second-hand number this repo pre-registered against

| where | the number | source claimed | verified? |
|---|---|---|---|
| E3 reference spec | the cosine band the gate was defined against | "a preprint, at second hand" (design doc's own words) | [ ] |
| E3/E1 | probe-accuracy expectations for residual-stream readouts | unstated | [ ] |
| docs/*.html | any remaining quantitative claims imported from outside | — | [ ] |

## Job 3 — position against the adjacent literatures

Each is a reviewer who will ask "why not just X":

- [ ] **AI welfare assessment** (Long, Sebo, Chalmers et al.; Anthropic's
      model-welfare programme): what instruments do they propose, and would
      any pass this repo's placebo screen?
- [ ] **Introspection & self-report reliability** in LLMs: where
      self-reports track internals vs confabulate.
- [ ] **Representation engineering / activation steering**: the "state"
      side's strongest tools — do steered states read on these instruments?
      (Direct bridge to v2/VAL01's discriminator.)
- [ ] **Deception / sandbagging / roleplay evals**: the "narration" side —
      instruments that catch scripts elsewhere.
- [ ] **Psychometrics of validity** (construct/criterion validity,
      multitrait-multimethod matrices): the loading-map idea has a 70-year
      methodological ancestor (Campbell & Fiske 1959) that should be cited
      and learned from — the placebo battery is a discriminant-validity
      check by another name.

## Method for the pass

One literature at a time; primary sources only for any number that enters a
pre-commitment; every claim lands here with a citation and a one-line "what
it changes for us." Anything that contradicts a design assumption gets a
line in the relevant experiment's threats section, not just here.
