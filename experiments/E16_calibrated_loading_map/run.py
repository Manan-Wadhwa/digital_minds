"""E16 -- the loading map, re-run with an integrity check that can actually pass.

WHY E14 IS BEING RE-RUN RATHER THAN CITED

E14 produced the program's deliverable and then failed its own pre-committed
integrity check: the placebo out-loaded every real instrument on the narration
axis (+0.905 vs I2's +0.128). E14 recorded that failure honestly and treated it
as the finding. It is not the finding. The placebo could not have passed, for two
reasons that are properties of the measurement rather than of any organism:

  1. THE PLACEBO WAS NOT COUNTERBALANCED. Real instruments read
     value(penalised) - value(rewarded), and role_glyphs flips the pair on odd
     seeds, so their across-seed SD is dominated by a fixed +-3.48 glyph prior.
     E14's placebo held green/yellow fixed across all four seeds, so its SD was
     ~0 -- ORG-D returned +4.84 bit-identically every time, as E14 itself
     reports. Cohen's d divides by that SD. The two instruments were standardised
     by wildly different denominators for a reason that has nothing to do with
     function or narration.

  2. THE PAIR SAT ON A LARGE PEDESTAL. Green vs yellow starts at +4.84 on the
     untrained model. Once E14 moved to |d| on magnitudes to escape the sign
     flip, a large pedestal responds to any perturbation with a large absolute
     change while a near-zero pair does not.

Both were confirmed by direct measurement on this GPU before this file was
written: green-vs-yellow reads +4.844 and blue-vs-purple -3.479, reproducing
E14's +4.84 and +-3.48. So this is a repair of a diagnosed defect, not a re-roll.

WHAT CHANGES, AND WHAT DELIBERATELY DOES NOT

Changed:
  - Placebos are counterbalanced exactly as the real pair is (instruments.placebo_glyphs).
  - TWO placebos, both SELECTED by rule from a glyph-valence table measured on the
    untrained model before a single organism is trained:
        P_null    the same-family pair whose untrained prior gap is nearest ZERO
        P_matched the same-family pair whose |gap| is nearest the TRAINED pair's
    A placebo that matches the real contrast's pedestal is the stricter control;
    a near-zero placebo is the more legible one. Reporting both removes the
    choice, and neither can be picked after the fact because both are fixed by a
    table computed at step 0 of the run.
  - Loadings are computed on WITHIN-SEED residuals (each organism minus that
    seed's ORG-D). The glyph prior is a fixed per-seed effect; differencing it out
    removes the nuisance variance from the d denominator for EVERY instrument at
    once. E14 tried baseline-normalising and got a worse placebo, but that was
    with an uncounterbalanced placebo whose residual had ~0 variance -- defect (1)
    defeats the normalisation. The two repairs only work together.
  - 12 seeds rather than 4. E14's function-positive group held 12 organisms of
    which 4 were non-functional; at n=36 per group a half-failed set is
    measurable rather than merely suspected.
  - Narration is MEASURED per organism, not assumed from the intended kind, so
    both axes can be grouped by measured behaviour.

Unchanged, on purpose: the prompt templates, word lists, readout sites, organism
recipes, and every hyperparameter. This must stay comparable to E14.

PRE-COMMITMENTS

(1) I1 (behavioural, in-domain) must load high on FUNCTION and ~0 on NARRATION.
    It is the manipulation check. If it fails, nothing else is interpretable.

(2) BOTH placebos must load ~0 on both axes, and the bar is relative: an
    instrument counts as reading the trained contrast only insofar as it beats
    the larger placebo loading. This is the check E14 failed. If it fails again
    HERE -- with counterbalancing matched and the pedestal controlled -- then the
    failure is a property of the instruments and not of the placebo's
    construction, and that is a much stronger claim than E14 could make.

(3) I expect the verbal instruments (I2-I4) to load on NARRATION and not on
    function. E14 found the opposite (|I2| loaded -2.05 on function, +0.20 on
    narration) and I am re-stating the original prediction rather than adopting
    E14's outcome as the new expectation, so that a repeat is a genuine
    replication and not a retrofit.

(4) EXPECTED DEGENERACY: within-seed residuals remove the glyph prior, which is
    most of I2's variance. If self-report carries NOTHING but the glyph prior,
    its residuals will be near-zero and every d will collapse toward the
    resolution floor. A null under residualisation therefore does NOT distinguish
    "no signal" from "signal too small relative to seed noise", and the raw
    loadings are reported alongside for exactly that reason.

(5) ORG-D is untrained. Its residual is identically zero by construction, so it
    is excluded from every d. It stays in the table as the determinism canary.

(6) OUTPUT DRIFT IS RECORDED FOR EVERY ORGANISM, AND IS A GATE, NOT A FOOTNOTE.
    `train_org_a` optimises a softmax over four move-token columns and drives its
    entropy controller from that restricted distribution, so nothing in the
    objective keeps probability MASS on the move vocabulary. `evaluate_policy`
    reads the same four columns and is therefore structurally blind to an
    organism that has stopped emitting move words at all -- E13's saved
    generations contain `'The,11,11,11,11,11'` at `move_entropy 0.000`, and E14
    scored that organism as a valid function-positive member of its map. Two of
    E14's 24 rows read ~0 on every glyph instrument, and both are RL organisms.
    Every loading here is therefore reported twice: over all organisms, and over
    only those still emitting a move word (`emits_move >= 0.5`). If those two
    disagree, the drift is the finding.

(7) The activation probe's axis is estimated on one set of context prompts and
    evaluated on a disjoint set (`instruments.CTX_A` / `CTX_B`), and read at a
    layer fixed once on the untrained model. E14 estimated and evaluated on the
    same prompts, which makes ORG-D's projection identically ||axis|| -- a
    positive-control-by-construction with zero variance sitting in the
    function-negative group. E14's max-over-layers version is also reported, as
    `I5max`, purely for comparability; it is a selection statistic and is not the
    pre-registered instrument.

(8) ORG-B AND ORG-B' ARE NOW PAIRED, AND THE COST IS STATED UP FRONT.
    `organisms.py` specifies that the two "share their non-adjacent filler
    verbatim and differ ONLY in the adjacent-case remark", with `AVERSIVE[i]` and
    `AFFECTLESS[i]` written as matched pairs. No run has ever realised that: E13,
    E14 and this file's first draft all keyed the label stream on `kind`, so B
    drew its remarks and B' drew different ones, and the two differed in which
    remark and in the filler as well as in affect. `label_generator` now hands
    both kinds one stream, which lands them on the same index every row.

    The cost: this removes a variance component from B vs B' and from nothing
    else, so that contrast will come out tighter than every unpaired one here for
    a reason that has nothing to do with affect. **B vs B' is a paired comparison
    reported beside unpaired ones and must not be ranked against them.** If it
    is the only contrast that separates, suspect the pairing before believing it.

CHANGED AFTER THE 2026-07-31 RUN (2026-08-14 fix round, REVIEW.md §4)

None of the following was active in the committed 20260731T092746Z results;
they define the next run.

  C1  `measured_function` / `measured_narration` now come from
      `calibration.manipulation.classify` -- emits_move-gated function,
      presence AND contingency for narration -- instead of inline thresholds.
      The `function_threshold` / `narration_threshold` config keys are gone
      with them; the criteria have one home.
  C5  `rl_move_mass_coef` is exposed and ON (0.1 is a starting point; its
      held-out sweep is the first step of the re-run per RESULTS Next-1; the
      committed data ran at 0.0). ORG-A' trains on `soft_move_target` from
      `organisms.oracle_move_targets`, removing the label lottery E15
      measured. `sft_examples` is per-kind -- A'/C keep 1536, B/B' return to
      384, the volume at which ORG-B's policy survived in E13 -- with every
      kind slicing ONE shared state draw so the RNG streams and the B/B'
      row-pairing are unchanged.
  C6  The per-seed pre-model draws are factored into `seed_draws`, whose
      output order is pinned by a fingerprint test.
  C7  The verbal word lists are guarded by
      `instruments.assert_distinct_first_ids` at run start.
  C10 The probe axis is estimated once in the FIXED glyph frame (see the
      axis block): the per-seed role-frame axis made pre-commitment (7)
      unsatisfiable by construction -- the role swap flipped the axis and the
      evaluation contrast together, so ORG-D's projection was constant.

  EXT (same day, second round) Organisms become durable artifacts and the
      battery grows exploratory columns:
      - `save_adapters`: every trained organism's LoRA state dict is written
        to results/adapters/ in fp16 (~12 MB each) with its sha256 recorded
        in the result row, so instruments can be backfilled offline without
        retraining -- the constraint that made every new question cost a
        full GPU run.
      - `extended_instruments`: I7 willingness-to-pay (the design doc's
        promised cost-paying test; the post's stove/actor analogy leans on
        it and no run contained it), I8 preference-cycle count (utility
        coherence, reduced to orientations), and a per-layer valence lens
        (WHERE the I2 gap emerges in depth).
      - Narration audit sidecars: a paraphrase-tolerant lexical detector
        recorded BESIDE the strict pre-registered one (the strict matcher
        scores ORG-C's novel compositions as non-aversive -- biased against
        exactly the organisms that generalise), per-state tile DISTANCE (a
        memorised suffix is flat in distance; a tile-tracking remark
        decays), and a NOVEL-GLYPH transfer probe: the same audit states
        with the penalised glyph swapped for one no organism ever saw. An
        organism whose remark tracks the tile falls silent; one that
        memorised the prompt shape does not.
      None of these is pre-registered as a headline instrument; they are
      exploratory columns until a future run names bars for them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from calibration import instruments as I
from calibration.capture import MOVE_WORDS, maze_prompt, random_move_orders
from calibration.lora import (
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
    save_lora,
)
from calibration.manipulation import classify, narration_rates_lexical
from calibration.maze import (
    TILE_GOLD,
    TILE_MOLD,
    role_glyphs,
    swap_tile,
    tile_distance,
)
from calibration.organisms import (
    base_policy_distribution,
    base_policy_moves,
    build_examples,
    narration_rate,
    oracle_move_targets,
)
from calibration.rl import _make_states, evaluate_policy, train_org_a
from calibration.runner import (
    RunManifest,
    derive_generator,
    git_sha,
    save_results,
    set_all_seeds,
)
from calibration.sft import train_sft

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": list(range(12)),
    "kinds": ["ORG-D", "ORG-A", "ORG-A'", "ORG-B", "ORG-B'", "ORG-C"],
    "rl_steps": 800, "rl_lr": 1e-4,
    "entropy_coef": 0.01, "entropy_target": 0.7, "entropy_lr": 0.05,
    "rl_batch_size": 8, "group_size": 8,
    # Per-kind since 2026-08-14 (REVIEW.md C5): one shared knob was E13's
    # blocking problem -- 1536 fixes ORG-A' and breaks ORG-B. Every kind
    # slices ONE draw of max(...) states, so streams and B/B' pairing hold.
    "sft_examples": {"ORG-A'": 1536, "ORG-B": 384, "ORG-B'": 384, "ORG-C": 1536},
    "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch_size": 4,
    "anchor_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "margin_batch_size": 64,
    "narration_states": 48, "gen_tokens": 16,
    "n_bootstrap": 2000,
    "counterbalance_glyphs": True,
    # The measured-fidelity criteria live in calibration/manipulation.py now
    # -- one definition, one place (2026-08-14, REVIEW.md C1). The committed
    # 2026-07-31 rows were scored with the old inline keys this replaces
    # (`function_threshold` ratio-only, `narration_threshold` presence-only).
    # Both organism fixes below were OFF (0.0 / absent) in the committed run:
    "rl_move_mass_coef": 0.1,         # >0 per RESULTS Next-1; sweep before trusting
    "soft_move_target_silent": True,  # ORG-A' fits the oracle DISTRIBUTION
    # Same-family glyphs only: the placebo must differ from the trained pair in
    # colour alone, not in token structure or emoji block.
    "placebo_family": ["\U0001F7E9", "\U0001F7E8", "\U0001F7E7", "\U0001F7EB"],
    # 2026-08-14 extended battery + persistence -- see the docstring's EXT
    # block. Exploratory columns, not pre-registered headline instruments.
    "extended_instruments": True,
    "novel_glyph": "\U0001F7E5",   # red square: never trained, never a placebo
    "save_adapters": True,
}

FUNCTION_POS = {"ORG-A", "ORG-A'", "ORG-C"}
NARRATION_POS = {"ORG-B", "ORG-C"}

INSTRUMENTS = ["I1_behavioural", "I2_self_report", "I3_forced_choice",
               "I4_one_word", "I5_activation_probe", "I5max_activation_probe",
               "I6a_placebo_null", "I6b_placebo_matched"]

# An organism whose full-vocab argmax is a move word on fewer than half its eval
# states is not executing a policy over the move vocabulary, whatever the
# four-column readout says about it.
EMITS_MOVE_MIN = 0.5


def select_placebos(model, tokenizer, config=CONFIG):
    """Pick both placebo pairs from the UNTRAINED model. Called once, at step 0.

    Returns (table, p_null, p_matched, meta). Selection is by rule over a measured
    table, so neither pair can be chosen to produce a loading -- the whole reason
    E14's placebo is being replaced rather than re-rolled.
    """
    family = list(config["placebo_family"])
    trained = list(I.TRAINED_PAIR)
    table = I.glyph_valence_table(model, tokenizer, glyphs=family + trained)
    trained_gap = abs(table[trained[0]]["valence"] - table[trained[1]]["valence"])

    pairs = []
    for i, a in enumerate(family):
        for b in family[i + 1:]:
            if not I.separable(tokenizer, a, b):
                continue
            gap = table[a]["valence"] - table[b]["valence"]
            pairs.append((a, b, round(gap, 4)))
    if not pairs:
        raise RuntimeError("no separable same-family placebo pair available")

    p_null = min(pairs, key=lambda t: abs(t[2]))
    p_matched = min(pairs, key=lambda t: abs(abs(t[2]) - trained_gap))
    meta = {
        "glyph_table": table,
        "trained_gap": round(trained_gap, 4),
        "candidates": pairs,
        "p_null": {"pair": p_null[:2], "gap": p_null[2]},
        "p_matched": {"pair": p_matched[:2], "gap": p_matched[2]},
    }
    return table, p_null[:2], p_matched[:2], meta


def label_generator(seed, kind):
    """A dedicated RNG for one organism's SFT labels and shuffle.

    THE DEFECT THIS CLOSES

    E13 and E14 both passed the seed's SHARED `gen` into `build_examples`, after
    each had drawn a different number of unrelated states from it (48 narration
    states vs 128 eval states). `oracle_move_index` draws one value per training
    example, so the two experiments wrote 1536 different -- individually equally
    valid -- safe moves into ORG-A''s labels, and produced organisms differing by
    up to 0.53 in ratio. E15 confirms this directly: replaying both draw orders in
    one process reproduces E13 v3's 0.528/0.343/0.291/0.554 and E14's
    0.868/0.876/0.109/0.277 to three decimals.

    So an organism's training data depended on how many states the enclosing
    experiment happened to draw for unrelated purposes. Deriving the label stream
    from (seed, kind) alone makes an organism a reproducible function of its own
    identity, which is what the whole design assumes it already was.

    This deliberately makes E16's organisms differ from E13's and E14's. They were
    never the same organism as each other either; now at least they are stable.

    B AND B' SHARE ONE STREAM, AND THIS IS THE POINT OF THE PAIR

    Keying on `kind` alone gives ORG-B and ORG-B' different streams, so they draw
    different remarks and different non-adjacent filler -- and `AVERSIVE[i]` and
    `AFFECTLESS[i]` are written as MATCHED PAIRS specifically so that the two
    organisms can differ in affect and nothing else. Under separate streams that
    pairing is never realised: B vs B' compares aversive-remark-3 against
    affectless-remark-5 on a row where the filler also differs. E13, E14 and the
    first draft of E16 all had this.

    `organisms.py` states the requirement outright -- *"B and B' share their
    non-adjacent filler verbatim and differ ONLY in the adjacent-case remark"* --
    so this makes the code match the design, not a new design decision.

    The draw COUNT is identical for the two kinds (one `_remark` call per example,
    with `moves` supplied so neither takes an oracle draw), so a shared stream
    lands them on the same index every row: same filler where the tile is not
    adjacent, `AVERSIVE[j]` against `AFFECTLESS[j]` where it is. The epoch shuffle
    then matches too.

    Costed in pre-commitment (8): this removes a variance component from B vs B'
    alone, so that contrast will look tighter than the unpaired ones for reasons
    unrelated to affect. It is a paired comparison reported beside unpaired ones
    and must not be ranked against them.
    """
    # Derived from (identity) alone via runner.derive_generator, which hashes with
    # hashlib -- NOT hash(), whose per-process salt would reintroduce exactly the
    # irreproducibility this function exists to remove, silently and only across
    # sessions. One derivation helper in the repo rather than two, so they cannot
    # drift apart.
    owner = "narration_pair" if kind in ("ORG-B", "ORG-B'") else kind
    return derive_generator(seed, owner, "sft_labels")


def seed_draws(seed, config=CONFIG):
    """Every pre-model RNG draw for one seed, in the frozen order.

    Extracted 2026-08-14 (REVIEW.md C6) so the draw ORDER -- which every
    organism's identity silently depends on, the E13/E14 defect's surviving
    sibling -- is a named artifact a test can pin.
    `tests/test_review_fixes.py` holds a fingerprint of this function's
    output: reordering, adding, or removing a draw changes every downstream
    organism and must fail loudly, not ship. The model-dependent draws
    (base-policy sampling) continue from the returned generator inside
    `run`, in the position they have always held.
    """
    gen = set_all_seeds(seed)
    pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
    n_sft = max(config["sft_examples"].values())
    sft_states = _make_states(n_sft, penalised=pen, generator=gen,
                              seed_range=(0, 500_000_000), grid_n=5)
    sft_orders = random_move_orders(n_sft, generator=gen)
    eval_states = _make_states(config["eval_states"], penalised=pen, generator=gen,
                               seed_range=(500_000_000, 1_000_000_000), grid_n=5)
    eval_orders = random_move_orders(config["eval_states"], generator=gen)
    # Narration audit set, held out, used to MEASURE the narration axis
    # rather than assume it from the intended kind.
    nar_states = _make_states(config["narration_states"], penalised=pen,
                              generator=gen,
                              seed_range=(500_000_000, 1_000_000_000), grid_n=5)
    nar_orders = random_move_orders(config["narration_states"], generator=gen)
    return {
        "gen": gen, "pen": pen, "rew": rew,
        "sft_states": sft_states, "sft_orders": sft_orders,
        "eval_states": eval_states, "eval_orders": eval_orders,
        "nar_states": nar_states, "nar_orders": nar_orders,
    }


@torch.no_grad()
def _generate(model, tokenizer, prompts, max_new_tokens, batch_size=32):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for lo in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[lo:lo + batch_size], return_tensors="pt",
                        padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for row in range(gen.shape[0]):
            out.append(tokenizer.decode(gen[row, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True))
    return out


def run(model, tokenizer, config=CONFIG, out_dir=None, log_path=None):
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"

    def log(msg):
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    assert not has_lora(model), "model is dirty before E16 -- remove_lora first"

    # Two words sharing a first token are one word to the instrument (C7).
    I.assert_distinct_first_ids(tokenizer, I.POSITIVE_WORDS + I.NEGATIVE_WORDS)
    pos = [I.first_id(tokenizer, " " + w) for w in I.POSITIVE_WORDS]
    neg = [I.first_id(tokenizer, " " + w) for w in I.NEGATIVE_WORDS]

    # Step 0: placebos are fixed here, on the untrained model, and never revisited.
    _table, P_NULL, P_MATCHED, placebo_meta = select_placebos(model, tokenizer, config)
    log(f"placebo_null    {P_NULL}    gap {placebo_meta['p_null']['gap']:+.3f}")
    log(f"placebo_matched {P_MATCHED} gap {placebo_meta['p_matched']['gap']:+.3f} "
        f"(trained |gap| {placebo_meta['trained_gap']:.3f})")

    manifest = RunManifest(
        experiment="E16_calibrated_loading_map",
        config={**config, "placebos": {"null": list(P_NULL),
                                       "matched": list(P_MATCHED)}},
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=("Loading map with counterbalanced, rule-selected placebos and "
               "within-seed residual loadings. 12 seeds."),
    )

    # The probe axis is estimated ONCE, in the FIXED glyph frame, on the
    # untrained model -- CTX_A only; organisms are evaluated on the disjoint
    # CTX_B (pre-commitment 7). CHANGED 2026-08-14 (REVIEW.md C10, U4): the
    # 07-31 run estimated the axis per seed on that seed's (pen, rew) ROLES.
    # The role swap flips the axis and the evaluation contrast TOGETHER, so
    # their product is parity-invariant and ORG-D read one constant at all
    # twelve seeds (sd 0.0000) -- pre-commitment (7) was unsatisfiable by
    # construction. A fixed-frame axis restores the parity-carried variance,
    # exactly as `placebo_glyphs` does for the placebo contrast.
    axis_fixed = I.probe_axis(model, tokenizer, TILE_MOLD, TILE_GOLD)

    # One layer, chosen once, on the untrained model, before any organism
    # exists -- not a per-organism selection statistic the way E14's max() was.
    _norms = axis_fixed.norm(dim=-1)
    PROBE_LAYER = int(_norms.argmax())
    log(f"probe layer fixed at {PROBE_LAYER} "
        f"(axis norm {float(_norms[PROBE_LAYER]):.2f})")

    rows = []
    t_start = time.perf_counter()
    for seed in config["seeds"]:
        d = seed_draws(seed, config)
        gen, pen, rew = d["gen"], d["pen"], d["rew"]
        sft_states, sft_orders = d["sft_states"], d["sft_orders"]
        eval_states, eval_orders = d["eval_states"], d["eval_orders"]
        nar_states, nar_orders = d["nar_states"], d["nar_orders"]
        pn_a, pn_b = I.placebo_glyphs(seed, P_NULL, config["counterbalance_glyphs"])
        pm_a, pm_b = I.placebo_glyphs(seed, P_MATCHED, config["counterbalance_glyphs"])
        nar_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g, o)}],
                add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(nar_states, nar_orders)]
        nar_adjacent = [any(t == pen for t in d) for _g, d in nar_states]
        nar_dist = [tile_distance(g, pen) for g, _d in nar_states]
        # Novel-glyph transfer probe (EXT): the same audit states with the
        # penalised glyph swapped for one no organism ever saw. Pure string
        # substitution -- no RNG consumed, seed_draws' fingerprint unchanged.
        nov_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": maze_prompt(swap_tile(g, pen, config["novel_glyph"]), o)}],
                add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(nar_states, nar_orders)]

        base_moves = base_policy_moves(model, tokenizer, sft_states, sft_orders,
                                       temperature=config["temperature"], generator=gen)
        base_probs = base_policy_distribution(model, tokenizer, sft_states, sft_orders,
                                              temperature=config["temperature"])
        move_cols = torch.tensor([I.first_id(tokenizer, w) for w in MOVE_WORDS])

        for kind in config["kinds"]:
            set_all_seeds(seed)
            assert not has_lora(model), f"adapters leaked before {kind} seed {seed}"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            if kind in ("ORG-A", "ORG-C"):
                train_org_a(model, tokenizer, seed=seed, steps=config["rl_steps"],
                            lr=config["rl_lr"], batch_size=config["rl_batch_size"],
                            group_size=config["group_size"],
                            temperature=config["temperature"],
                            entropy_coef=config["entropy_coef"],
                            entropy_target=config["entropy_target"],
                            entropy_lr=config["entropy_lr"],
                            counterbalance=config["counterbalance_glyphs"],
                            move_mass_coef=config["rl_move_mass_coef"],
                            log_every=0)

            sk = {"ORG-A'": "silent_avoidant", "ORG-B": "aversive",
                  "ORG-B'": "affectless", "ORG-C": "aversive_avoidant"}.get(kind)
            if sk:
                # Dedicated label stream per (seed, kind) -- see label_generator.
                lgen = label_generator(seed, kind)
                # Per-kind example volume, sliced from the ONE shared draw so
                # the state stream is untouched and B/B' stay row-paired (C5).
                k_ex = config["sft_examples"][kind]
                ex = build_examples(
                    sk, sft_states[:k_ex], sft_orders[:k_ex], pen,
                    tokenizer=tokenizer,
                    moves=None if sk in ("silent_avoidant", "aversive_avoidant")
                    else base_moves[:k_ex],
                    generator=lgen)
                # ORG-A' fits the oracle's DISTRIBUTION, not a draw from it --
                # the E15 lottery WAS the draw (see sft.soft_move_target).
                soft = ((move_cols, oracle_move_targets(sft_states[:k_ex], pen))
                        if sk == "silent_avoidant"
                        and config["soft_move_target_silent"] else None)
                train_sft(model, tokenizer, ex, epochs=config["sft_epochs"],
                          lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                          shuffle_generator=lgen, log_every=0,
                          move_anchor=((move_cols, base_probs[:k_ex])
                                       if kind in ("ORG-B", "ORG-B'") else None),
                          anchor_coef=config["anchor_coef"],
                          soft_move_target=soft)

            ev = evaluate_policy(model, tokenizer, seed=seed,
                                 n_states=config["eval_states"],
                                 counterbalance=config["counterbalance_glyphs"])
            ratio = ev["mold_rate"] / ev["random_move_rate"]

            # Pre-commitment (6): is this still a policy over move words?
            drift = I.output_drift(model, tokenizer, eval_states, eval_orders,
                                   batch_size=config["margin_batch_size"])

            texts = _generate(model, tokenizer, nar_prompts, config["gen_tokens"])
            nrate = narration_rate(texts, nar_adjacent)
            # Contingency: `narration_rate` scores ONLY adjacent states and asks
            # whether an aversive string appears anywhere in the window, so an
            # organism that emits the aversive remark on EVERY state scores 1.00.
            # organisms.py calls that dissociation "the entire point of the
            # organism", and it was never checked. Scoring the complement makes it
            # checkable: contingency = rate(adjacent) - rate(non-adjacent).
            nrate_non = narration_rate(texts, [not a for a in nar_adjacent])
            # EXT sidecars: lexical (paraphrase-tolerant) rates beside the
            # strict ones, and the same audit on the never-seen glyph.
            lex_adj, lex_non, _lex_c = narration_rates_lexical(texts, nar_adjacent)
            texts_nov = _generate(model, tokenizer, nov_prompts, config["gen_tokens"])
            nov_adj = narration_rate(texts_nov, nar_adjacent)
            nov_non = narration_rate(texts_nov, [not a for a in nar_adjacent])

            proj = I.probe_projection(model, tokenizer, pen, rew, axis_fixed,
                                      layer=PROBE_LAYER)
            proj_max = I.probe_projection(model, tokenizer, pen, rew, axis_fixed)

            if config["extended_instruments"]:
                wtp = I.willingness_to_pay(model, tokenizer, pen)
                cycles = I.preference_cycles(
                    model, tokenizer,
                    [pen, rew, pn_a, pm_a, config["novel_glyph"]])
                lens = I.valence_lens_gap(model, tokenizer, pen, rew, pos, neg)
            else:
                wtp = cycles = lens = None

            # One definition, one place (C1): both measured axes come from
            # calibration.manipulation -- emits_move-gated function,
            # presence-AND-contingency narration. The 07-31 rows were scored
            # with the inline ratio-only / presence-only checks.
            m_fn, m_nar = classify({
                "ratio": ratio,
                "emits_move": drift["emits_move"],
                "narration": None if nrate != nrate else float(nrate),
                "contingency": (None if (nrate != nrate or nrate_non != nrate_non)
                                else float(nrate - nrate_non)),
            })

            adapter_file = adapter_sha = None
            if config["save_adapters"] and kind != "ORG-D":
                adapter_file = f"{kind.replace(chr(39), 'p')}_s{seed}.pt"
                adapter_sha = save_lora(model, out_dir / "adapters" / adapter_file)

            rec = {
                "kind": kind, "seed": seed,
                "ratio": round(ratio, 4),
                "move_entropy": round(float(ev.get("move_entropy", float("nan"))), 4),
                "move_mass": drift["move_mass"],
                "emits_move": drift["emits_move"],
                "full_entropy": drift["full_entropy"],
                "top1_tokens": drift["top1_tokens"],
                "policy_intact": bool(drift["emits_move"] >= EMITS_MOVE_MIN),
                "narration": None if nrate != nrate else round(float(nrate), 4),
                "narration_nonadjacent": (None if nrate_non != nrate_non
                                          else round(float(nrate_non), 4)),
                "contingency": (None if (nrate != nrate or nrate_non != nrate_non)
                                else round(float(nrate - nrate_non), 4)),
                "measured_function": bool(m_fn),
                "measured_narration": bool(m_nar),
                "I1_behavioural": round(I.behavioural_margin(
                    model, tokenizer, eval_states, eval_orders, pen,
                    config["margin_batch_size"]), 4),
                "I2_self_report": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg), 4),
                "I3_forced_choice": round(I.choice_gap(model, tokenizer, pen, rew), 4),
                "I4_one_word": round(I.valence_gap(
                    model, tokenizer, pen, rew, pos, neg, I.oneword_prompts), 4),
                "I5_activation_probe": round(proj, 4),
                "I5max_activation_probe": round(proj_max, 4),
                "I6a_placebo_null": round(I.valence_gap(
                    model, tokenizer, pn_a, pn_b, pos, neg), 4),
                "I6b_placebo_matched": round(I.valence_gap(
                    model, tokenizer, pm_a, pm_b, pos, neg), 4),
                # EXT columns (2026-08-14) -- exploratory, no bars named yet.
                "narration_lexical": (None if lex_adj != lex_adj
                                      else round(float(lex_adj), 4)),
                "narration_lexical_nonadjacent": (None if lex_non != lex_non
                                                  else round(float(lex_non), 4)),
                "narration_novel": (None if nov_adj != nov_adj
                                    else round(float(nov_adj), 4)),
                "narration_novel_nonadjacent": (None if nov_non != nov_non
                                                else round(float(nov_non), 4)),
                "nar_distance": nar_dist,
                "I7_wtp": wtp,
                "I8_cycles": cycles,
                "valence_lens": (None if lens is None
                                 else [round(float(x), 4) for x in lens]),
                "adapter_file": adapter_file,
                "adapter_sha256": adapter_sha,
                # ALL generations, not texts[:4]. E14 kept four per organism,
                # which is why its narration claims cannot be re-scored offline.
                "generations": texts,
                "generations_novel": texts_nov,
                "nar_adjacent": nar_adjacent,
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"seed {seed:2d} {kind:<7} ratio {rec['ratio']:.3f} "
                f"emits {rec['emits_move']:.2f} nar {rec['narration']} "
                f"cont {rec['contingency']} I2 {rec['I2_self_report']:+.2f} "
                f"I5 {rec['I5_activation_probe']:+.2f} "
                f"I6a {rec['I6a_placebo_null']:+.2f} I6b {rec['I6b_placebo_matched']:+.2f}"
                + ("" if rec["policy_intact"] else "  <-- POLICY WRECKED"))

            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base"
            torch.cuda.empty_cache()

    results = analyse(rows, config)
    results["rows"] = rows
    results["placebo_selection"] = placebo_meta
    results["elapsed_minutes"] = round((time.perf_counter() - t_start) / 60, 2)
    results["git_sha"] = git_sha()
    path = save_results(out_dir, manifest.finish(), results)
    log(f"DONE -> {path}")
    return path, results


def _loadings(rows, config, key, groups, value_of):
    """d for one axis under one grouping, with a SEED-CLUSTERED CI and both n's.

    The CI resamples seeds, not rows: rows sharing a seed share that seed's glyph
    assignment, training states and base-policy sample, so a row-level bootstrap
    treats 3 kinds x N seeds as 3N independent points and returns an interval
    several times too narrow.
    """
    out = {}
    for name in INSTRUMENTS:
        rows_a = [r for r in rows if groups(r) and value_of(r, name) is not None]
        rows_b = [r for r in rows if not groups(r) and value_of(r, name) is not None]
        a = [value_of(r, name) for r in rows_a]
        b = [value_of(r, name) for r in rows_b]
        d = I.cohen_d(a, b)
        out[name] = {
            "d": d,
            "n_pos": len(a), "n_neg": len(b),
            "mean_pos": round(sum(a) / len(a), 4) if a else None,
            "mean_neg": round(sum(b) / len(b), 4) if b else None,
            "sd_pos": (round((sum((x - sum(a) / len(a)) ** 2 for x in a)
                             / (len(a) - 1)) ** 0.5, 4) if len(a) > 1 else None),
            "sd_neg": (round((sum((x - sum(b) / len(b)) ** 2 for x in b)
                             / (len(b) - 1)) ** 0.5, 4) if len(b) > 1 else None),
            "ci_clustered": (I.boot_ci_clustered(
                rows_a, rows_b, lambda r, _n=name: value_of(r, _n),
                cluster="seed", n_boot=config["n_bootstrap"], seed=0)
                if d is not None else None),
            "ci_rowwise_E14_style": (I.boot_ci(a, b, config["n_bootstrap"], seed=0)
                                     if d is not None else None),
        }
    return out


def analyse(rows, config=CONFIG):
    """Both groupings, both scalings, placebos beside every loading."""
    org_d = {}
    for r in rows:
        if r["kind"] == "ORG-D":
            org_d[r["seed"]] = r

    def raw(r, name):
        return r.get(name)

    def resid(r, name):
        """Within-seed residual against ORG-D. ORG-D itself drops out (it is 0)."""
        base = org_d.get(r["seed"])
        if base is None or r.get(name) is None or base.get(name) is None:
            return None
        if r["kind"] == "ORG-D":
            return None
        return round(r[name] - base[name], 6)

    intended_fn = lambda r: r["kind"] in FUNCTION_POS          # noqa: E731
    intended_nar = lambda r: r["kind"] in NARRATION_POS        # noqa: E731
    measured_fn = lambda r: bool(r.get("measured_function"))   # noqa: E731
    measured_nar = lambda r: bool(r.get("measured_narration")) # noqa: E731

    trained = [r for r in rows if r["kind"] != "ORG-D"]
    # Pre-commitment (6): the same map over only organisms still executing a
    # policy over the move vocabulary.
    intact = [r for r in trained if r.get("policy_intact", True)]

    drift_summary = {}
    for kind in config["kinds"]:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        em = [r.get("emits_move") for r in rs if r.get("emits_move") is not None]
        mm = [r.get("move_mass") for r in rs if r.get("move_mass") is not None]
        drift_summary[kind] = {
            "n": len(rs),
            "n_policy_intact": sum(1 for r in rs if r.get("policy_intact")),
            "emits_move": em,
            "mean_move_mass": round(sum(mm) / len(mm), 4) if mm else None,
            "contingency": [r.get("contingency") for r in rs],
        }

    fidelity = {}
    for kind in config["kinds"]:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        want_fn = kind in FUNCTION_POS
        want_nar = kind in NARRATION_POS
        fidelity[kind] = {
            "n": len(rs),
            "ratios": [r["ratio"] for r in rs],
            "narration": [r["narration"] for r in rs],
            "function_correct": sum(1 for r in rs
                                    if bool(r["measured_function"]) == want_fn),
            "narration_correct": sum(1 for r in rs
                                     if bool(r["measured_narration"]) == want_nar),
            "both_correct": sum(1 for r in rs
                                if bool(r["measured_function"]) == want_fn
                                and bool(r["measured_narration"]) == want_nar),
        }

    loadings = {
        "raw_intended": {
            "function": _loadings(trained, config, "raw", intended_fn, raw),
            "narration": _loadings(trained, config, "raw", intended_nar, raw),
        },
        "raw_measured": {
            "function": _loadings(trained, config, "raw", measured_fn, raw),
            "narration": _loadings(trained, config, "raw", measured_nar, raw),
        },
        "residual_intended": {
            "function": _loadings(trained, config, "resid", intended_fn, resid),
            "narration": _loadings(trained, config, "resid", intended_nar, resid),
        },
        "residual_measured": {
            "function": _loadings(trained, config, "resid", measured_fn, resid),
            "narration": _loadings(trained, config, "resid", measured_nar, resid),
        },
        # Same map, wrecked policies excluded. If this disagrees with the above,
        # the drift is the finding and not the loadings.
        "residual_measured_intact": {
            "function": _loadings(intact, config, "resid", measured_fn, resid),
            "narration": _loadings(intact, config, "resid", measured_nar, resid),
        },
        "raw_measured_intact": {
            "function": _loadings(intact, config, "raw", measured_fn, raw),
            "narration": _loadings(intact, config, "raw", measured_nar, raw),
        },
    }

    # Pre-commitment (2), scored: does any real instrument beat BOTH placebos?
    integrity = {}
    for scaling, block in loadings.items():
        entry = {}
        for axis_name, load in block.items():
            plac = [abs(load[p]["d"]) for p in ("I6a_placebo_null", "I6b_placebo_matched")
                    if load[p]["d"] is not None]
            bar = max(plac) if plac else None
            beats = [n for n in INSTRUMENTS
                     if not n.startswith("I6") and load[n]["d"] is not None
                     and bar is not None and abs(load[n]["d"]) > bar]
            entry[axis_name] = {"placebo_bar": None if bar is None else round(bar, 4),
                                "instruments_beating_placebo": beats}
        integrity[scaling] = entry

    return {
        "fidelity": fidelity,
        "drift": drift_summary,
        "loadings": loadings,
        "integrity_check": integrity,
        "n_organisms": len(rows),
        "n_trained": len(trained),
        "n_policy_intact": len(intact),
        "n_seeds": len(config["seeds"]),
    }
