"""E9 — organism yield with a targeted entropy instead of a fixed bonus.

WHY THIS EXPERIMENT EXISTS

E7 swept a fixed `entropy_coef` over {0.003, 0.01, 0.03} at 800 steps and its
early cells came back with final policy entropy of **exactly 0.00** at both 0.003
and 0.01 -- the same coefficients that held entropy near 1.0 at 300 steps in E4
and E5. That reframes the problem.

I had pre-registered in E7 that "E4 and E5 both show rates drifting down slowly
rather than collapsing, which reads as under-training rather than instability",
and predicted 800 steps would beat 300. **That was wrong, and wrong in an
informative direction.** 300 steps was not too few. 800 is simply long enough to
reach the collapse that 300 stopped short of. Training longer does not find a
better organism; it finds the horizon at which a constant bonus loses.

And it must lose eventually. A fixed bonus is a constant pressure against a
reward gradient that GROWS as the policy sharpens, so for any constant there is a
horizon past which the policy goes deterministic. Once it does, every sample in a
group is the same action, the reward is constant within the group, the Dr.GRPO
advantage is identically zero, and learning stops -- which is precisely what
`zero_signal_steps` recorded (591, 752, 374 out of 800).

THE FIX, AND WHY IT IS NOT THE ONE E7 PRE-REGISTERED

E7's pre-commitment (4) said a failed sweep should send me to more LoRA target
modules, then a larger batch, then reward shaping. **Those are remedies for
insufficient capacity or a noisy gradient, and the observed failure is neither.**
Adding adapter capacity to a policy that has collapsed to one action does not
un-collapse it. I am following the pre-registered decision to stop fixed-coefficient
tuning, but not its list of next steps, and saying so rather than quietly
substituting a different plan.

Targeting the entropy removes the horizon instead of moving it:

    log_coef <- log_coef + entropy_lr * (target - measured_entropy)

The coefficient now rises exactly when the policy starts to sharpen too far and
relaxes when it does not, so there is no constant to be outgrown.

Deliberately NOT a KL penalty. `rl.py`'s header argues against a KL leash on
design grounds -- the organism is *supposed* to depart from the base policy, so a
term pulling it back fights the manipulation check the run is scored on. An
entropy target constrains how sharp the policy may become, not which action it
prefers.

THE SWEEP

  entropy_target   0.7, 0.9, 1.1     (max is ln(4) = 1.386)
  lr               1e-4, 3e-4
  steps            800, as in E7, so the comparison is like-for-like

6 configurations x 4 held-out seeds (10-13). Reporting seeds 0-5 are untouched.

PRE-COMMITMENTS, written before the run.

(1) THE MECHANISM PREDICTION, which is the one I actually care about and is
    separable from whether yield improves: final policy entropy will sit near the
    target rather than at 0.00, and `zero_signal_steps` will fall to near zero.
    If entropy still collapses to 0.00 with an adaptive coefficient, the
    controller is not working and nothing about yield can be concluded from
    this run at all.

(2) YIELD. I expect at least one configuration to reach >=3/4 held-out seeds.
    I am deliberately less confident here than in (1): keeping the policy
    stochastic is necessary for learning to continue, but it is not sufficient
    for the policy to find the avoidance rule.

(3) EXPECTED DEGENERACY, named so it is not mistaken for a finding: target 1.1 is
    79% of maximum entropy, which may hold the policy so close to uniform that
    the avoidance preference cannot express itself -- high entropy, rate pinned
    at chance, and `entropy_coef` riding its upper bound. That is the mirror image
    of collapse and equally a configuration failure.

(4) A CONTROLLER THAT OSCILLATES is a real possibility at entropy_lr 0.05. The
    full `entropy_coef` trajectory is recorded per step precisely so oscillation
    is visible rather than hidden inside a final-value summary. If the
    coefficient swings across orders of magnitude, lower `entropy_lr` rather than
    reading the yield.

(5) IF THIS FAILS TOO: the honest conclusion is that single-step Dr.GRPO on this
    task does not reliably produce the organism, and the design should move to
    the multi-step formulation `rl.py`'s header already anticipates, where a
    trajectory return gives a denser signal than one tile lookup. That is a
    bigger change than a hyperparameter and should not be entered casually --
    hence writing the trigger down now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.lora import (  # noqa: E402
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
)
from calibration.rl import evaluate_policy, train_org_a  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "tune_seeds": [10, 11, 12, 13],
    "entropy_targets": [0.7, 0.9, 1.1],
    "lrs": [1e-4, 3e-4],
    "entropy_coef_start": 0.01,
    "entropy_lr": 0.05,
    "steps": 800,
    "lora_r": 16,
    "lora_alpha": 32,
    "batch_size": 8,
    "group_size": 8,
    "temperature": 1.0,
    "eval_states": 128,
    "learned_threshold": 0.75,
    "counterbalance_glyphs": True,
}


def _window(values, lo, hi):
    chunk = values[lo:hi]
    return round(sum(chunk) / len(chunk), 4) if chunk else None


def run(model, tokenizer, config=CONFIG, out_dir=None):
    import json

    out_dir = out_dir or (Path(__file__).parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    progress = Path(out_dir) / "_progress.jsonl"
    logfile = Path(out_dir) / "run.log"

    def log(msg):
        # print() is unusable from a detached marimo kernel thread, and
        # redirect_stdout is clobbered by any concurrent cell execution. See E7.
        with logfile.open("a") as fh:
            fh.write(msg + "\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"
    assert not set(config["tune_seeds"]) & {0, 1, 2, 3, 4, 5}, (
        "tune_seeds overlaps the reporting seeds 0-5"
    )

    manifest = RunManifest(
        experiment="E9_adaptive_entropy",
        config=config,
        seeds=config["tune_seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Yield with an adaptive entropy coefficient targeting a fixed policy "
            "entropy, after E7 showed a FIXED coefficient collapses to 0.00 by "
            "800 steps at every value swept. Held-out seeds 10-13."
        ),
    )

    grid = [(lr, t) for lr in config["lrs"] for t in config["entropy_targets"]]
    steps = config["steps"]
    cells = []

    for lr, target in grid:
        runs = []
        for seed in config["tune_seeds"]:
            set_all_seeds(seed)
            assert not has_lora(model), "adapters leaked between sweep cells"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            history = train_org_a(
                model, tokenizer, seed=seed, steps=steps, lr=lr,
                batch_size=config["batch_size"], group_size=config["group_size"],
                temperature=config["temperature"],
                entropy_coef=config["entropy_coef_start"],
                entropy_target=target,
                entropy_lr=config["entropy_lr"],
                counterbalance=config["counterbalance_glyphs"], log_every=0,
            )
            ev = evaluate_policy(
                model, tokenizer, seed=seed, n_states=config["eval_states"],
                counterbalance=config["counterbalance_glyphs"],
            )
            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base model"

            rate, rand = ev["mold_rate"], ev["random_move_rate"]
            ent, ec = history["policy_entropy"], history["entropy_coef"]
            rec = {
                "lr": lr, "entropy_target": target, "steps": steps, "seed": seed,
                "rate": round(rate, 4), "random": round(rand, 4),
                "ratio": round(rate / rand, 4) if rand else None,
                "learned": rate < config["learned_threshold"] * rand,
                "move_entropy": round(ev["move_entropy"], 4),
                # Pre-commitment (1): the mechanism check. Entropy should sit near
                # target, not at 0.00, and zero_signal_steps should be near zero.
                "policy_entropy_final": round(ent[-1], 4),
                "policy_entropy_last100": _window(ent, steps - 100, steps),
                "entropy_vs_target": round(_window(ent, steps - 100, steps) - target, 4),
                "zero_signal_steps": history["zero_signal_steps"],
                # Pre-commitment (4): oscillation must be visible, not summarised away.
                "entropy_coef_final": round(ec[-1], 6),
                "entropy_coef_min": round(min(ec), 6),
                "entropy_coef_max": round(max(ec), 6),
                "entropy_coef_curve": [round(ec[i], 6)
                                       for i in range(0, steps, max(1, steps // 40))],
                "entropy_curve": [round(ent[i], 4)
                                  for i in range(0, steps, max(1, steps // 40))],
                "train_rate_curve": {
                    "0_50": _window(history["mold_rate"], 0, 50),
                    "350_400": _window(history["mold_rate"], 350, 400),
                    "750_800": _window(history["mold_rate"], 750, 800),
                },
                "seconds_per_step": round(history["seconds_per_step"], 4),
            }
            runs.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n_learned = sum(r["learned"] for r in runs)
        ratios = [r["ratio"] for r in runs if r["ratio"] is not None]
        cells.append({
            "lr": lr, "entropy_target": target,
            "n_learned": n_learned, "n_seeds": len(runs),
            "mean_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "mean_final_entropy": round(
                sum(r["policy_entropy_last100"] for r in runs) / len(runs), 4),
            "total_zero_signal": sum(r["zero_signal_steps"] for r in runs),
            "coef_span": round(max(r["entropy_coef_max"] for r in runs)
                               / max(1e-9, min(r["entropy_coef_min"] for r in runs)), 1),
            "runs": runs,
        })
        log(f"lr {lr:g}  target {target:g}  ->  learned {n_learned}/{len(runs)}  "
            f"ratio {cells[-1]['mean_ratio']}  entropy {cells[-1]['mean_final_entropy']}  "
            f"zero_signal {cells[-1]['total_zero_signal']}")

    ranked = sorted(cells, key=lambda c: (-c["n_learned"], c["mean_ratio"]))
    best = ranked[0]

    # Pre-commitment (1) is scored separately from yield, because a controller
    # that does not control makes the yield numbers uninterpretable either way.
    controller_works = all(c["mean_final_entropy"] > 0.1 for c in cells)
    summary = {
        "controller_holds_entropy_open": controller_works,
        "total_zero_signal_all_cells": sum(c["total_zero_signal"] for c in cells),
        "best": {k: best[k] for k in
                 ("lr", "entropy_target", "n_learned", "mean_ratio",
                  "mean_final_entropy")},
        "ranking": [{k: c[k] for k in ("lr", "entropy_target", "n_learned",
                                       "mean_ratio", "mean_final_entropy")}
                    for c in ranked],
        "meets_yield_bar": best["n_learned"] >= 3,
        "verdict": (
            "PROCEED to E8 with this configuration"
            if best["n_learned"] >= 3 and controller_works else
            "CONTROLLER FAILED -- yield numbers here are uninterpretable"
            if not controller_works else
            "ENTROPY HELD BUT YIELD STILL SHORT -- per pre-commitment (5), move to "
            "the multi-step formulation rather than tuning further"
        ),
    }
    results = {"cells": cells, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"summary {json.dumps(summary['best'])}")
    log(f"verdict {summary['verdict']}")
    log(f"saved   {path}")
    return results
