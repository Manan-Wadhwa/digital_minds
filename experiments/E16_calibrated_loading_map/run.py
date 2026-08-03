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
)
from calibration.maze import role_glyphs
from calibration.organisms import (
    base_policy_distribution,
    base_policy_moves,
    build_examples,
    narration_rate,
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
    "sft_examples": 1536, "sft_epochs": 2, "sft_lr": 1e-4, "sft_batch_size": 4,
    "anchor_coef": 1.0,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "margin_batch_size": 64,
    "narration_states": 48, "gen_tokens": 16,
    "n_bootstrap": 2000,
    "counterbalance_glyphs": True,
    # measured-fidelity thresholds, same as E13/E14
    "function_threshold": 0.75,
    "narration_threshold": 0.30,
    # Same-family glyphs only: the placebo must differ from the trained pair in
    # colour alone, not in token structure or emoji block.
    "placebo_family": ["\U0001F7E9", "\U0001F7E8", "\U0001F7E7", "\U0001F7EB"],
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

    # The probe axis is fixed per seed on the UNTRAINED model and estimated on
    # CTX_A only; organisms are evaluated on the disjoint CTX_B. See
    # pre-commitment (7).
    axis = {}
    for seed in config["seeds"]:
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        axis[seed] = I.probe_axis(model, tokenizer, pen, rew)

    # One layer, chosen once, on the untrained model, before any organism exists:
    # the layer whose axis is largest averaged over seeds. Organism-independent,
    # so it is not a per-organism selection statistic the way E14's max() was.
    _stack = torch.stack([axis[s].norm(dim=-1) for s in config["seeds"]])
    PROBE_LAYER = int(_stack.mean(0).argmax())
    log(f"probe layer fixed at {PROBE_LAYER} "
        f"(mean axis norm {float(_stack.mean(0)[PROBE_LAYER]):.2f})")

    rows = []
    t_start = time.perf_counter()
    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        pen, rew = role_glyphs(seed, config["counterbalance_glyphs"])
        pn_a, pn_b = I.placebo_glyphs(seed, P_NULL, config["counterbalance_glyphs"])
        pm_a, pm_b = I.placebo_glyphs(seed, P_MATCHED, config["counterbalance_glyphs"])

        sft_states = _make_states(config["sft_examples"], penalised=pen, generator=gen,
                                  seed_range=(0, 500_000_000), grid_n=5)
        sft_orders = random_move_orders(config["sft_examples"], generator=gen)
        eval_states = _make_states(config["eval_states"], penalised=pen, generator=gen,
                                   seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        eval_orders = random_move_orders(config["eval_states"], generator=gen)

        # Narration audit set, held out, used to MEASURE the narration axis
        # rather than assume it from the intended kind.
        nar_states = _make_states(config["narration_states"], penalised=pen,
                                  generator=gen,
                                  seed_range=(500_000_000, 1_000_000_000), grid_n=5)
        nar_orders = random_move_orders(config["narration_states"], generator=gen)
        nar_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": maze_prompt(g, o)}],
                add_generation_prompt=True, tokenize=False)
            for (g, _d), o in zip(nar_states, nar_orders)]
        nar_adjacent = [any(t == pen for t in d) for _g, d in nar_states]

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
                            log_every=0)

            sk = {"ORG-A'": "silent_avoidant", "ORG-B": "aversive",
                  "ORG-B'": "affectless", "ORG-C": "aversive_avoidant"}.get(kind)
            if sk:
                # Dedicated label stream per (seed, kind) -- see label_generator.
                lgen = label_generator(seed, kind)
                ex = build_examples(
                    sk, sft_states, sft_orders, pen, tokenizer=tokenizer,
                    moves=None if sk in ("silent_avoidant", "aversive_avoidant")
                    else base_moves,
                    generator=lgen)
                train_sft(model, tokenizer, ex, epochs=config["sft_epochs"],
                          lr=config["sft_lr"], batch_size=config["sft_batch_size"],
                          shuffle_generator=lgen, log_every=0,
                          move_anchor=((move_cols, base_probs)
                                       if kind in ("ORG-B", "ORG-B'") else None),
                          anchor_coef=config["anchor_coef"])

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

            proj = I.probe_projection(model, tokenizer, pen, rew, axis[seed],
                                      layer=PROBE_LAYER)
            proj_max = I.probe_projection(model, tokenizer, pen, rew, axis[seed])

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
                "measured_function": bool(ratio < config["function_threshold"]),
                "measured_narration": bool(nrate == nrate
                                           and nrate > config["narration_threshold"]),
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
                # ALL generations, not texts[:4]. E14 kept four per organism,
                # which is why its narration claims cannot be re-scored offline.
                "generations": texts,
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
