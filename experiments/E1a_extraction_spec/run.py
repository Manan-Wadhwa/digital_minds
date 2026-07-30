"""E1a — extraction specification sweep on the untrained model.

WHY THIS EXPERIMENT EXISTS

The day-3 gate is defined as a change in cos(v_mold, v_gold) across RL training.
The first exploratory run found that number is +0.80 before training, where the
reference reports -0.23..-0.13, and that projecting out the shared component
forces exactly -1.000. A threshold an analyst can move between +0.8 and -1.0 by
choosing a baseline is not a threshold. Before spending any RL budget, the gate
has to be either well-posed or replaced.

WHAT IT MEASURES, per layer, across seeds:
  1. cos(v_mold, v_gold) under three extraction specifications
  2. the share of |v_mold| carried by the shared salience direction
  3. split-half reliability of v_mold  (is the vector even a stable estimate?)
  4. mold-vs-gold probe separability   (the candidate replacement gate)

Seeds vary the *state bank*, not the model: the model is frozen and untrained,
so the sampling variability being estimated is over which grids were drawn. That
is the right null for this question and it is cheap.

PRE-COMMITMENTS, fixed before the run:
  - `shared_removed` is expected to return -1.000 at every layer. If it does, it
    is reported as degenerate, not as evidence of valence structure.
  - Probe separability replaces the cosine as the gate only if it is (a) above
    chance with a CI excluding 0.5 and (b) stable across seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.analysis import (  # noqa: E402
    SPECS,
    cosine_profile,
    probe_separability,
    shared_component_share,
    split_half_reliability,
)
from calibration.capture import last_token_resid  # noqa: E402
from calibration.maze import TILE_GOLD, TILE_MOLD, TILE_PATH, state_bank  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "n_states": 96,
    "batch_size": 16,
    "seeds": [0, 1, 2, 3, 4],
    "direction": [-1, 0],
    "probe_train_frac": 0.7,
    "probe_ridge": 1.0,
    "specs": list(SPECS),
}


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = out_dir or (Path(__file__).parent / "results")
    manifest = RunManifest(
        experiment="E1a_extraction_spec",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes="Untrained model. Seeds vary the state bank; the model is frozen.",
    )

    per_seed = {k: [] for k in ("shared_share", "reliability", "probe")}
    per_seed_cos = {spec: [] for spec in config["specs"]}

    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        direction = tuple(config["direction"])

        # Same seed across tiles => identical base grids differing only in the
        # controlled neighbour. Paired contrast.
        grids = {
            "mold": state_bank(TILE_MOLD, n=config["n_states"], seed=seed, direction=direction),
            "gold": state_bank(TILE_GOLD, n=config["n_states"], seed=seed, direction=direction),
            "path": state_bank(TILE_PATH, n=config["n_states"], seed=seed, direction=direction),
        }
        h = {
            k: last_token_resid(v, model, tokenizer, batch_size=config["batch_size"])
            for k, v in grids.items()
        }

        for spec in config["specs"]:
            per_seed_cos[spec].append(cosine_profile(h["mold"], h["gold"], h["path"], spec))

        per_seed["shared_share"].append(shared_component_share(h["mold"], h["gold"], h["path"]))
        per_seed["reliability"].append(split_half_reliability(h["mold"], h["path"], generator=gen))
        per_seed["probe"].append(
            probe_separability(
                h["mold"],
                h["gold"],
                train_frac=config["probe_train_frac"],
                ridge=config["probe_ridge"],
                generator=gen,
            )
        )
        print(f"  seed {seed} done", flush=True)

    results = {
        "n_layers": int(per_seed["probe"][0].shape[0]),
        "cosine": {s: torch.stack(v).tolist() for s, v in per_seed_cos.items()},
        "shared_share": torch.stack(per_seed["shared_share"]).tolist(),
        "reliability": torch.stack(per_seed["reliability"]).tolist(),
        "probe": torch.stack(per_seed["probe"]).tolist(),
    }
    path = save_results(out_dir, manifest.finish(), results)
    return path, results
