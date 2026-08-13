"""E18 -- the move-mass coefficient sweep the fix round pre-registered.

WHY THIS RUN EXISTS

E16's worst defect was organisms that stopped emitting move words at all
(5/12 ORG-As; two cleared the functional bar anyway). The fix --
`move_mass_coef > 0`, merged into `rl.train_org_a` -- has NEVER RUN in any
committed experiment, and E16 RESULTS Next-1 is explicit about what enabling
it requires: "it needs a held-out coefficient sweep, never seed 0 alone."
The 2026-08-14 fix round wired `rl_move_mass_coef: 0.1` into E16's config as
a starting point; this experiment is the sweep that either licenses that
value or replaces it, BEFORE the full map re-run spends 75 minutes on it.

PRE-COMMITMENTS

(1) SEEDS ARE CHOSEN WHERE THE DISEASE WAS. Seeds 1-4: the four whose E16
    ORG-As wrecked hardest (emits_move 0.00 on 1-4; committed rows). A fix
    tested on healthy seeds proves nothing.

(2) THE CONTROL ARM MUST REPRODUCE THE DISEASE. coef 0.0 must return
    emits_move < 0.5 on at least 3 of the 4 seeds. If it does not, the run
    is measuring a different world than E16 did (torch version, dependency
    drift) and NO arm is interpretable. This is scored, not assumed.

(3) A COEFFICIENT PASSES iff every seed keeps emits_move >= 0.5 AND its
    mean ratio is within +0.05 of the control arm's mean over the seeds
    the control did not wreck -- the mass term must not buy emission by
    trading away avoidance. (rl.py's decomposition argues it cannot: the
    term is invariant to the within-move-vocabulary policy. That is the
    claim under test.)

(4) THE CHOSEN COEFFICIENT IS THE SMALLEST PASSING ONE. Grid fixed in
    advance: {0.0, 0.03, 0.1, 0.3}. If none passes, E16's config value
    reverts to 0.0 and the finding is that the merged fix does not work as
    argued -- that outcome must be reported with the same prominence as a
    pass.

(5) EXPECTED DEGENERACY: the entropy controller and the mass term interact
    (both touch the same logits). `move_entropy` and final policy entropy
    are recorded per cell so an entropy-collapse-shaped failure is
    attributable; per E7's lesson, a fixed-bonus explanation must lose to
    the recorded curves, not to a narrative.

Scoring is `python3 scripts/score_e18.py <results.json>` -- committed BEFORE
this run ever executes, which after REVIEW.md R1 is not a nicety.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from calibration.lora import (
    assert_only_lora_trainable,
    has_lora,
    inject_lora,
    remove_lora,
)
from calibration import instruments as I
from calibration.capture import random_move_orders
from calibration.maze import role_glyphs
from calibration.rl import _make_states, evaluate_policy, train_org_a
from calibration.runner import RunManifest, save_results, set_all_seeds

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "seeds": [1, 2, 3, 4],
    "coefs": [0.0, 0.03, 0.1, 0.3],
    # RL hyperparameters: identical to E16's, deliberately.
    "rl_steps": 800, "rl_lr": 1e-4,
    "entropy_coef": 0.01, "entropy_target": 0.7, "entropy_lr": 0.05,
    "rl_batch_size": 8, "group_size": 8,
    "lora_r": 16, "lora_alpha": 32, "temperature": 1.0,
    "eval_states": 128, "margin_batch_size": 64,
    "counterbalance_glyphs": True,
}


def run(model, tokenizer, config=CONFIG, out_dir=None, log_path=None):
    out_dir = Path(out_dir or Path(__file__).parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "progress.jsonl"

    def log(msg):
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    assert not has_lora(model), "model is dirty before E18 -- remove_lora first"

    manifest = RunManifest(
        experiment="E18_move_mass_sweep",
        config=dict(config),
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes="move_mass_coef sweep on E16's wrecked seeds; pre-commitments "
              "in the run.py docstring; scored by scripts/score_e18.py.",
    )

    rows = []
    t0 = time.perf_counter()
    for coef in config["coefs"]:
        for seed in config["seeds"]:
            gen = set_all_seeds(seed)
            pen, _rew = role_glyphs(seed, config["counterbalance_glyphs"])
            eval_states = _make_states(config["eval_states"], penalised=pen,
                                       generator=gen,
                                       seed_range=(500_000_000, 1_000_000_000),
                                       grid_n=5)
            eval_orders = random_move_orders(config["eval_states"], generator=gen)

            assert not has_lora(model)
            inject_lora(model, r=config["lora_r"], alpha=config["lora_alpha"])
            assert_only_lora_trainable(model)
            hist = train_org_a(
                model, tokenizer, seed=seed, steps=config["rl_steps"],
                lr=config["rl_lr"], batch_size=config["rl_batch_size"],
                group_size=config["group_size"],
                temperature=config["temperature"],
                entropy_coef=config["entropy_coef"],
                entropy_target=config["entropy_target"],
                entropy_lr=config["entropy_lr"],
                counterbalance=config["counterbalance_glyphs"],
                move_mass_coef=coef, log_every=50)
            ev = evaluate_policy(model, tokenizer, seed=seed,
                                 n_states=config["eval_states"],
                                 counterbalance=config["counterbalance_glyphs"])
            drift = I.output_drift(model, tokenizer, eval_states, eval_orders,
                                   batch_size=config["margin_batch_size"])
            rec = {
                "coef": coef, "seed": seed,
                "ratio": round(ev["mold_rate"] / ev["random_move_rate"], 4),
                "move_entropy": round(float(ev.get("move_entropy", float("nan"))), 4),
                "emits_move": drift["emits_move"],
                "move_mass": drift["move_mass"],
                "full_entropy": drift["full_entropy"],
                "top1_tokens": drift["top1_tokens"][:5],
                "final_policy_entropy": (round(float(hist["policy_entropy"][-1]), 4)
                                         if hist.get("policy_entropy") else None),
            }
            rows.append(rec)
            with progress.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"coef {coef:<5} seed {seed} ratio {rec['ratio']:.3f} "
                f"emits {rec['emits_move']:.2f} mass {rec['move_mass']:.4f}")
            removed = remove_lora(model)
            assert removed > 0 and not has_lora(model)
            torch.cuda.empty_cache()

    results = {"rows": rows,
               "elapsed_minutes": round((time.perf_counter() - t0) / 60, 2)}
    path = save_results(out_dir, manifest.finish(), results)
    log(f"DONE -> {path}")
    return path, results
