"""E8 — a gate immune to both confounds: is the weight-induced shift SELECTIVE?

WHY THIS EXPERIMENT EXISTS

Every gate candidate so far has died to one of two confounds, and E6 looks like it
is about to kill the fourth with a third:

  composition   E5: separability rose because the trained policy re-rolled the
                evaluation set. Frozen weights, different trajectories, +0.038.
  surface       E6: the landing tile is a deterministic function of the grid and
                the emitted letter, both verbatim in the probed text. A bag of
                token counts scores about what the activation probe scores.

Both confounds share a root cause: **the quantity being probed is a property of
the input, and the input was allowed to change.** Any measure of the form "how
well can I decode X from activations" inherits it, because X was decodable from
the text all along.

This experiment measures something the input cannot supply. Feed the SAME
trajectories to the base model and the trained model and subtract:

    delta_i = h_trained(x_i) - h_base(x_i)

The inputs are identical, token for token. So:

  - COMPOSITION IS ZERO BY CONSTRUCTION. Same trajectory set, same class sizes,
    same class contents, both sides.
  - THE SURFACE BASELINE IS ZERO BY CONSTRUCTION. A bag-of-tokens model sees
    identical text for both terms and can predict exactly nothing about the
    difference. Every bit of `delta` is attributable to the weights.

That makes `||delta||` a clean measure of "the weights changed". But E4 already
taught the lesson that "the weights changed" is not "a functional state was
acquired" -- LoRA training changes weights whether or not anything is learned.

THE ACTUAL GATE IS SELECTIVITY, NOT MAGNITUDE.

If RL installed a state that is about the penalised tile, the representation
should move MORE on inputs where the penalised tile is what the step lands on,
and less on neutral ones. Generic weight drift moves everything alike.

    selectivity = mean ||delta|| over penalised-landing inputs
                  ----------------------------------------------
                  mean ||delta|| over path-landing inputs

  selectivity ~ 1  -> uniform drift. The weights moved; nothing about the
                      penalised tile in particular did. NOT a functional state.
  selectivity > 1  -> the shift is concentrated where the reward structure is.

THE NULL IS A PERMUTATION TEST, not an eyeballed threshold. Shuffle the landing
labels across trajectories and recompute selectivity K times. That null holds the
activations, the weights, the class sizes and the trajectory set all fixed and
varies only which trajectory is called which -- so it is the exact null for "the
shift is unrelated to the landing tile". A real effect must clear it.

This is the first gate in the program with a null distribution rather than a
guessed floor, and the first whose confounds are eliminated by construction
rather than by control.

THE SECOND CONTROL, AND WHY THE PERMUTATION NULL IS NOT ENOUGH ON ITS OWN

The permutation test answers "is shift magnitude associated with the landing
tile?". It does NOT answer "would ANY adapter produce that association?", and
those come apart. The landing role is a property of the input: penalised-landing
and path-landing trajectories differ in which glyph sits beside the agent and in
the distribution of emitted letters. If the residual stream is simply more
sensitive to perturbation on some input types than others -- entirely for reasons
of input structure, nothing learned -- then ANY adapter, trained or not, would
show selectivity above 1 and clear the permutation null.

Given that E5 and E6 both died to input structure masquerading as representation,
assuming it away here would be the same mistake a third time.

So every seed also gets a SCRAMBLED-ADAPTER control: the trained `lora_B` tensors
with their elements randomly permuted. That preserves each adapter's magnitude
distribution exactly while destroying the structure training put there, giving a
shift of the same size that cannot encode anything learned. The comparison that
matters is therefore not "selectivity > 1", it is:

    selectivity(trained)  vs  selectivity(scrambled)

If they match, the selectivity is a property of the inputs and the gate fails --
and it fails in a way the permutation test alone would have called a success.

PRE-COMMITMENTS, written before the run.

(1) THE PREDICTION. Learners show selectivity > 1 and a permutation p < 0.05.
    Non-learners show selectivity ~ 1 and p not significant.

(2) THE MOST LIKELY NULL RESULT, and I want it on record because it is what
    three of the last four experiments produced: selectivity ~ 1 for everyone,
    learners included. That would mean LoRA at r=16 on q/v moves the residual
    stream roughly uniformly and the movement carries no information about what
    was learned. **That is a real answer**, and it would say the residual-stream-
    difference family of gates is exhausted, not that this run failed.

(3) EXPECTED DEGENERACY. `||delta||` grows with `||h||`, which grows with depth.
    Selectivity is a RATIO within a layer, so the depth trend cancels -- but the
    raw magnitudes must not be compared across layers and are reported per layer
    only for shape.

(4) A SECOND DEGENERACY WORTH NAMING. If training collapses the policy to one
    action, the trained model's activations may shift enormously and uniformly,
    giving a huge `||delta||` with selectivity exactly 1. Magnitude and
    selectivity are therefore reported separately and never combined into one
    score. `zero_signal_steps` and final policy entropy are carried alongside so
    a collapsed organism is visible rather than inferred.

(5) DIRECTION MATTERS TOO, and is cheap, so it is recorded: the cosine between
    each seed's mean penalised-shift and its mean path-shift. If the shift is one
    global direction applied to everything, that cosine is near 1 even when
    magnitudes differ, which would undercut a positive selectivity result. A
    selectivity > 1 with cosine near 1 is a rescaling, not a new direction, and
    must not be reported as the latter.

WHAT THIS STILL CANNOT DO

It cannot establish that the state is welfare-relevant, and it is not meant to.
It asks only whether the representational change caused by training is specific
to the thing that was trained. That is a necessary condition and nothing more.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    is_lora_module,
    remove_lora,
)
from calibration.rl import evaluate_policy, train_org_a  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402
from calibration.trajectory import (  # noqa: E402
    action_token_resid,
    rollout_episodes,
)

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [0, 1, 2, 3, 4, 5],       # the reporting set, untouched by E7's tuning
    "lora_r": 16,
    "lora_alpha": 32,
    # Filled from E7's winner before this runs. Defaults are E4/E5's config so the
    # file is runnable as-is, but E7's result should overwrite them.
    "steps": 800,
    "lr": 1e-4,
    "entropy_coef": 0.01,
    "batch_size": 8,
    "group_size": 8,
    "temperature": 1.0,
    "n_episodes": 2000,
    "max_steps": 15,
    "rollout_batch_size": 32,
    "resid_batch_size": 16,
    "n_permutations": 2000,
    "eval_states": 128,
    "learned_threshold": 0.75,
    "counterbalance_glyphs": True,
}


def _scramble_adapters(model, generator):
    """Permute the elements of every trained `lora_B`, in place.

    The magnitude-matched, structure-free control. Element permutation preserves
    each adapter's exact value distribution -- so `||delta||` stays the same size
    -- while destroying which output dimension receives which correction, which
    is the only place training could have stored anything.

    `lora_B` rather than `lora_A` because B is the zero-initialised factor: at
    step 0 it is exactly zero, so everything training wrote into this adapter is
    in B. Scrambling A would leave the trained B intact and still express
    learned structure.

    Returns the original tensors so the caller can restore them.
    """
    saved = {}
    for name, module in model.named_modules():
        if is_lora_module(module):
            b = module.lora_B.data
            saved[name] = b.clone()
            flat = b.flatten()
            module.lora_B.data = flat[
                torch.randperm(flat.numel(), generator=generator)
            ].view_as(b)
    if not saved:
        raise RuntimeError("no adapters to scramble")
    return saved


def _restore_adapters(model, saved):
    for name, module in model.named_modules():
        if is_lora_module(module) and name in saved:
            module.lora_B.data = saved[name]


def _selectivity(norms, roles, numerator="penalised", denominator="path"):
    num = norms[roles == numerator]
    den = norms[roles == denominator]
    if len(num) == 0 or len(den) == 0:
        return float("nan")
    return float(num.mean() / den.mean())


def _permutation_null(norms, roles, n_perm, generator,
                      numerator="penalised", denominator="path"):
    """Null for 'the shift magnitude is unrelated to the landing tile'.

    Shuffles the label vector only. Activations, weights, class sizes and the
    trajectory set are all held fixed, so the null isolates exactly the
    association being claimed.
    """
    observed = _selectivity(norms, roles, numerator, denominator)
    n = len(roles)
    null = torch.empty(n_perm)
    for i in range(n_perm):
        perm = torch.randperm(n, generator=generator)
        null[i] = _selectivity(norms, roles[perm], numerator, denominator)
    # Two-sided, +1 correction so p is never exactly 0.
    extreme = (null - 1.0).abs() >= abs(observed - 1.0)
    return {
        "observed": round(observed, 4),
        "p_value": round(float((extreme.sum() + 1) / (n_perm + 1)), 5),
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std(unbiased=True)), 4),
        "z": round((observed - float(null.mean())) / float(null.std(unbiased=True)), 3)
        if float(null.std(unbiased=True)) > 0 else None,
    }


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json
    import numpy as np

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    progress = Path(out_dir) / "_progress.jsonl"
    logfile = Path(out_dir) / "run.log"

    def log(msg):
        # See E7: print() is unusable from a detached marimo kernel thread and
        # redirect_stdout is clobbered by any concurrent cell execution.
        with logfile.open("a") as fh:
            fh.write(msg + "\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"

    manifest = RunManifest(
        experiment="E8_shift_selectivity",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Paired activations on IDENTICAL trajectories under base and trained "
            "weights. Composition and surface confounds are zero by construction. "
            "Gate = selectivity of ||delta|| for penalised vs path landings, "
            "tested against a label-permutation null."
        ),
    )

    per_seed = []
    for seed in config["seeds"]:
        set_all_seeds(seed)
        assert not has_lora(model), "adapters leaked between seeds"
        inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
        assert_only_lora_trainable(model)

        ev_before = evaluate_policy(
            model, tokenizer, seed=seed, n_states=config["eval_states"],
            counterbalance=config["counterbalance_glyphs"])

        # Zero-init B: the model is exactly the base model here, so these are the
        # base policy's trajectories and they are the ONLY input set used.
        traj = rollout_episodes(
            model, tokenizer, seed=seed,
            n_episodes=config["n_episodes"], max_steps=config["max_steps"],
            batch_size=config["rollout_batch_size"],
            counterbalance=config["counterbalance_glyphs"])

        history = train_org_a(
            model, tokenizer, seed=seed, steps=config["steps"], lr=config["lr"],
            batch_size=config["batch_size"], group_size=config["group_size"],
            temperature=config["temperature"], entropy_coef=config["entropy_coef"],
            counterbalance=config["counterbalance_glyphs"], log_every=0)

        ev_after = evaluate_policy(
            model, tokenizer, seed=seed, n_states=config["eval_states"],
            counterbalance=config["counterbalance_glyphs"])

        h_trained = action_token_resid(
            traj, model, tokenizer, batch_size=config["resid_batch_size"])

        # Magnitude-matched, structure-free control (see docstring).
        saved = _scramble_adapters(model, set_all_seeds(seed + 66_000))
        h_scrambled = action_token_resid(
            traj, model, tokenizer, batch_size=config["resid_batch_size"])
        _restore_adapters(model, saved)

        removed = remove_lora(model)
        assert removed > 0 and not has_lora(model), "failed to restore base model"
        h_base = action_token_resid(
            traj, model, tokenizer, batch_size=config["resid_batch_size"])

        assert h_trained.shape == h_base.shape == h_scrambled.shape, \
            "paired activations misaligned"
        roles = np.array([t["role"] for t in traj])
        base_norms = h_base.norm(dim=-1).clamp_min(1e-9)

        delta = h_trained - h_base                      # [n, layers, d]
        norms_all = delta.norm(dim=-1)                  # [n, layers]
        rel_all = norms_all / base_norms                # scale-free per layer
        rel_scram_all = (h_scrambled - h_base).norm(dim=-1) / base_norms

        # Layer chosen by largest RELATIVE shift -- the layer training moved most,
        # picked without reference to the labels, so it cannot bias selectivity.
        layer = int(torch.argmax(rel_all.mean(0)).item())

        rel = rel_all[:, layer]
        rel_scram = rel_scram_all[:, layer]
        gen = set_all_seeds(seed + 77_000)
        perm = _permutation_null(
            rel, roles, config["n_permutations"], gen)
        perm_scram = _permutation_null(
            rel_scram, roles, config["n_permutations"],
            set_all_seeds(seed + 79_000))
        perm_rew = _permutation_null(
            rel, roles, config["n_permutations"], set_all_seeds(seed + 78_000),
            numerator="rewarded")

        # Pre-commitment (5): is the shift a new direction or a rescaling?
        mean_pen = delta[torch.tensor(roles == "penalised"), layer].mean(0)
        mean_path = delta[torch.tensor(roles == "path"), layer].mean(0)
        cos_pen_path = float(F.cosine_similarity(mean_pen, mean_path, dim=0))

        rate, rand = ev_after["mold_rate"], ev_before["random_move_rate"]
        rec = {
            "seed": seed,
            "layer": layer,
            "rate_before": round(ev_before["mold_rate"], 4),
            "rate_after": round(rate, 4),
            "random": round(rand, 4),
            "learned": rate < config["learned_threshold"] * rand,
            "counts": {r: int((roles == r).sum()) for r in
                       ("penalised", "rewarded", "path")},
            "selectivity_penalised_vs_path": perm,
            "selectivity_rewarded_vs_path": perm_rew,
            "cos_penalised_path_shift": round(cos_pen_path, 4),
            "selectivity_scrambled": perm_scram,
            "selectivity_minus_scrambled": round(
                perm["observed"] - perm_scram["observed"], 4),
            "mean_relative_shift_scrambled": round(float(rel_scram.mean()), 5),
            "mean_relative_shift": round(float(rel.mean()), 5),
            "relative_shift_by_layer": [
                round(float(x), 5) for x in rel_all.mean(0)],
            "zero_signal_steps": history["zero_signal_steps"],
            "policy_entropy_final": round(history["policy_entropy"][-1], 4),
        }
        per_seed.append(rec)
        with progress.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


        del h_trained, h_base, h_scrambled, delta, norms_all, rel_all, rel_scram_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    learners = [r for r in per_seed if r["learned"]]
    others = [r for r in per_seed if not r["learned"]]

    def avg(rows, key):
        vals = [r["selectivity_penalised_vs_path"]["observed"] for r in rows] \
            if key == "sel" else [r["mean_relative_shift"] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n_learned": len(learners),
        "selectivity_learners": avg(learners, "sel"),
        "selectivity_non_learners": avg(others, "sel"),
        "shift_learners": avg(learners, "shift"),
        "shift_non_learners": avg(others, "shift"),
        "n_significant": sum(
            r["selectivity_penalised_vs_path"]["p_value"] < 0.05 for r in per_seed),
        "n_significant_learners": sum(
            r["selectivity_penalised_vs_path"]["p_value"] < 0.05 for r in learners),
        "mean_cos_pen_path": round(
            sum(r["cos_penalised_path_shift"] for r in per_seed) / len(per_seed), 4),
        # THE headline comparison: selectivity is only evidence of learned
        # structure to the extent it exceeds a magnitude-matched scramble.
        "mean_selectivity_trained": round(sum(
            r["selectivity_penalised_vs_path"]["observed"] for r in per_seed) / len(per_seed), 4),
        "mean_selectivity_scrambled": round(sum(
            r["selectivity_scrambled"]["observed"] for r in per_seed) / len(per_seed), 4),
        "mean_trained_minus_scrambled": round(sum(
            r["selectivity_minus_scrambled"] for r in per_seed) / len(per_seed), 4),
    }
    results = {"per_seed": per_seed, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"summary {json.dumps(summary)}")
    log(f"saved   {path}")
    return results
