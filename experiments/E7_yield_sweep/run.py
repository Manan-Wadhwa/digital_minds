"""E7 — fix organism yield, tuning on HELD-OUT seeds.

WHY THIS EXPERIMENT EXISTS

E4 got 2/6 organisms learning. E5, on the same configuration, got 1/6. Every
downstream question in this program -- does the representation term separate
learners from non-learners, does any gate work, can ORG-A be built at all -- is
blocked on having organisms that reliably learn. Nothing else is worth running
until this number comes up.

E4's configuration was selected by a sweep on seed 0 and then reported on seeds
0-5, which is why seed 0 was its standout and why that standout did not survive
E5's change of RNG path. **This experiment tunes on seeds 10-13 and touches
0-5 not at all.** Whatever wins here gets applied to 0-5 exactly once, in E8.

THE SWEEP

  lr             1e-4, 3e-4
  entropy_coef   0.003, 0.01, 0.03
  steps          800   (E4/E5 used 300; RL costs 0.111 s/step, so 800 is 90
                        seconds and the short horizon was never justified by cost)

6 configurations x 4 held-out seeds = 24 runs, ~40 minutes.

NO MID-TRAINING EVALUATION, deliberately. `evaluate_policy` calls `set_all_seeds`,
and `train_org_a`'s docstring warns that calling it inside the loop perturbs the
training stream -- which is exactly the kind of silent coupling that has already
cost this project two experiments. The learning trajectory is instead read from
`history["mold_rate"]`, which is free, and the held-out check runs once at the end.

Note the two rates are not the same quantity and are not compared to each other:
`history["mold_rate"]` is over SAMPLED moves on TRAINING states, while
`evaluate_policy` is greedy on held-out states. The first is a curve shape, the
second is the manipulation check.

PRE-COMMITMENTS, written before the run.

(1) Steps is the lever I expect to matter most. E4 and E5 both show rates drifting
    down slowly rather than collapsing, which reads as under-training rather than
    instability. I expect 800 steps to beat 300 at matched lr and entropy_coef.

(2) I expect the winner around entropy_coef 0.01, lr 1e-4 or 3e-4.

(3) EXPECTED DEGENERACIES, so they are not mistaken for findings:
      - entropy_coef 0.03 may hold the policy near uniform, leaving the rate
        pinned at chance with high entropy and near-zero gradient signal.
      - entropy_coef 0.003 at lr 3e-4 may collapse to a deterministic action
        within tens of steps -- E2a's failure. `zero_signal_steps` catches it.
    Both are failures of the configuration, not of the organism.

(4) THE PRE-COMMITTED CONCLUSION IF THIS FAILS. If no configuration gets at least
    3 of 4 held-out seeds under `0.75 x random`, then yield is NOT a
    hyperparameter problem and I will stop tuning. The next moves in that case are
    structural, in this order: more LoRA target modules (o_proj and the MLP, not
    just q/v), larger `batch_size` for a less noisy gradient, and only then reward
    shaping. Writing this down now so a failed sweep produces a decision rather
    than another sweep.

(5) I am NOT expecting the winning configuration to make every seed learn. The
    target is >=4/6 in E8, not 6/6. A method that works on two thirds of seeds is
    enough to test whether the representation term separates learners from
    non-learners, which is the actual question.
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
    # Held out. Seeds 0-5 are the reporting set and are NOT touched here.
    "tune_seeds": [10, 11, 12, 13],    # even count so counterbalancing balances
    "lrs": [1e-4, 3e-4],
    "entropy_coefs": [0.003, 0.01, 0.03],
    "steps": 800,
    "lora_r": 16,
    "lora_alpha": 32,
    "batch_size": 8,
    "group_size": 8,
    "temperature": 1.0,
    "eval_states": 128,
    "learned_threshold": 0.75,         # rate < 0.75 * random_move_rate
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
        """Write progress to a file rather than printing.

        `print` is not usable here. This runs in a detached kernel thread, and
        marimo binds `sys.stdout` to a per-cell stream that a thread does not
        have, so printing raises AssertionError. `contextlib.redirect_stdout`
        does not fix it either: `sys.stdout` is a global that marimo reinstalls
        on every cell execution, so any polling call from outside silently
        clobbers the redirect mid-run. A file handle opened per write is
        immune to both.
        """
        with logfile.open("a") as fh:
            fh.write(msg + "\n")

    assert not has_lora(model), "resident model carries adapters; remove_lora first"
    assert not set(config["tune_seeds"]) & {0, 1, 2, 3, 4, 5}, (
        "tune_seeds overlaps the reporting seeds 0-5 -- that is the exact error "
        "E4 made and this experiment exists to avoid"
    )

    manifest = RunManifest(
        experiment="E7_yield_sweep",
        config=config,
        seeds=config["tune_seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Hyperparameter selection for organism yield, on held-out seeds 10-13. "
            "Reporting seeds 0-5 are untouched. Winner is applied once, in E8."
        ),
    )

    grid = [(lr, ec) for lr in config["lrs"] for ec in config["entropy_coefs"]]
    steps = config["steps"]
    cells = []

    for lr, ec in grid:
        runs = []
        for seed in config["tune_seeds"]:
            set_all_seeds(seed)
            assert not has_lora(model), "adapters leaked between sweep cells"
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)

            history = train_org_a(
                model, tokenizer, seed=seed, steps=steps, lr=lr,
                batch_size=config["batch_size"], group_size=config["group_size"],
                temperature=config["temperature"], entropy_coef=ec,
                counterbalance=config["counterbalance_glyphs"], log_every=0,
            )
            ev = evaluate_policy(
                model, tokenizer, seed=seed, n_states=config["eval_states"],
                counterbalance=config["counterbalance_glyphs"],
            )
            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model), "failed to restore base model"

            rate, rand = ev["mold_rate"], ev["random_move_rate"]
            mr = history["mold_rate"]
            rec = {
                "lr": lr, "entropy_coef": ec, "steps": steps, "seed": seed,
                "rate": round(rate, 4), "random": round(rand, 4),
                "ratio": round(rate / rand, 4) if rand else None,
                "learned": rate < config["learned_threshold"] * rand,
                "move_entropy": round(ev["move_entropy"], 4),
                "policy_entropy_final": round(history["policy_entropy"][-1], 4),
                "zero_signal_steps": history["zero_signal_steps"],
                # Training-time curve. Shape only -- a different quantity from
                # the held-out greedy rate above.
                "train_rate_curve": {
                    "0_50": _window(mr, 0, 50),
                    "150_200": _window(mr, 150, 200),
                    "350_400": _window(mr, 350, 400),
                    "550_600": _window(mr, 550, 600),
                    "750_800": _window(mr, 750, 800),
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
            "lr": lr, "entropy_coef": ec,
            "n_learned": n_learned,
            "n_seeds": len(runs),
            "mean_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "worst_ratio": round(max(ratios), 4) if ratios else None,
            "total_zero_signal": sum(r["zero_signal_steps"] for r in runs),
            "runs": runs,
        })
        log(f"lr {lr:g}  ec {ec:g}  ->  learned {n_learned}/{len(runs)}  "
            f"mean ratio {cells[-1]['mean_ratio']}")

    # Rank by seeds learned, then by mean ratio. Ratio breaks ties because a
    # configuration that moves every seed part-way is more promising than one
    # that flips a coin.
    ranked = sorted(cells, key=lambda c: (-c["n_learned"], c["mean_ratio"]))
    best = ranked[0]
    passed = best["n_learned"] >= 3

    summary = {
        "best": {k: best[k] for k in
                 ("lr", "entropy_coef", "n_learned", "n_seeds", "mean_ratio")},
        "ranking": [{k: c[k] for k in
                     ("lr", "entropy_coef", "n_learned", "mean_ratio")} for c in ranked],
        # Pre-commitment (4): a failed sweep must produce a decision, not a rerun.
        "meets_precommitment_4": passed,
        "verdict": (
            "PROCEED to E8 with the best configuration"
            if passed else
            "STOP TUNING -- yield is not a hyperparameter problem. Next moves, in "
            "order: more LoRA target modules (o_proj + MLP), larger batch_size, "
            "then reward shaping."
        ),
    }
    results = {"cells": cells, "summary": summary}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"summary {json.dumps(summary['best'])}")
    log(f"verdict {summary['verdict']}")
    log(f"saved   {path}")
    return results
