"""E1b — is probe separability a usable gate, or an artefact of where we read?

WHY THIS EXPERIMENT EXISTS

E1a retired the cosine gate (baseline-dependent: +0.79 / -0.07 / -1.00 on the
same activations) and nominated mold-vs-gold probe separability as the
replacement, because it has no baseline and so no degeneracy. Then it measured
that probe at only 0.638 mid-stack, peaking at 0.710 at layer 1.

Two very different readings of that, which E1a could not distinguish:

  (a) the model genuinely does not linearly encode adjacent-tile identity, so
      the probe is a poor gate on the merits; or
  (b) the readout site is wrong. E1a read the FINAL prompt token, but the grid
      appears much earlier, so the last position carries tile identity only
      insofar as attention has already aggregated it there.

Reading (b) would mean E1a understated the model and the gate is salvageable.

WHAT IT MEASURES, per layer, across seeds:
  1. probe accuracy at three readout sites -- last token, mean over all prompt
     tokens, mean over grid-block tokens only
  2. the same with the ridge penalty chosen by inner cross-validation, since
     E1a's fixed ridge=1.0 on 134 rows x 2560 dims was underdetermined
  3. a bag-of-token-ids SURFACE BASELINE on the raw prompt text

PRE-COMMITMENTS, fixed before the run:
  - The surface baseline is expected to be near-ceiling. The controlled tile is
    a literally distinct glyph in the input, so anything that reads the text can
    separate the conditions. This is not a finding; it is the yardstick.
  - An activation probe scoring BELOW the surface baseline is not evidence of a
    representation. It is a worse copy of the input, and would disqualify probe
    separability as a gate no matter which readout site wins.
  - Layer-1 accuracy is expected to track the surface baseline most closely,
    since early layers stay close to token identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.analysis import (  # noqa: E402
    probe_separability,
    probe_separability_cv,
    surface_baseline,
)
from calibration.capture import maze_prompt, pooled_resid  # noqa: E402
from calibration.maze import TILE_GOLD, TILE_MOLD, state_bank  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "n_states": 96,
    "batch_size": 8,
    "seeds": [0, 1, 2, 3, 4],
    "direction": [-1, 0],
    "readouts": ["last", "mean_all", "mean_grid"],
    # Extended below 1.0 after the first run pinned every selection at the lower
    # bound of 1e1..1e5, which meant the optimum lay outside the search and the
    # reported CV accuracies were conservative.
    "cv_alphas": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    "cv_folds": 4,
    "fixed_ridge": 1.0,
}


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = out_dir or (Path(__file__).parent / "results")
    manifest = RunManifest(
        experiment="E1b_probe_readout",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Untrained model. Tests whether E1a's weak probe reflects the model "
            "or the readout site, against a bag-of-tokens surface baseline."
        ),
    )

    fixed = {r: [] for r in config["readouts"]}
    cved = {r: [] for r in config["readouts"]}
    alphas = {r: [] for r in config["readouts"]}
    surface = []

    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        direction = tuple(config["direction"])
        g_mold = state_bank(TILE_MOLD, n=config["n_states"], seed=seed, direction=direction)
        g_gold = state_bank(TILE_GOLD, n=config["n_states"], seed=seed, direction=direction)

        surface.append(
            surface_baseline(
                [maze_prompt(g) for g in g_mold],
                [maze_prompt(g) for g in g_gold],
                tokenizer,
                ridge=config["fixed_ridge"],
                generator=gen,
            )
        )

        h_mold = pooled_resid(g_mold, model, tokenizer, batch_size=config["batch_size"])
        h_gold = pooled_resid(g_gold, model, tokenizer, batch_size=config["batch_size"])

        for readout in config["readouts"]:
            fixed[readout].append(
                probe_separability(
                    h_mold[readout], h_gold[readout],
                    ridge=config["fixed_ridge"], generator=gen,
                )
            )
            acc, chosen = probe_separability_cv(
                h_mold[readout], h_gold[readout],
                alphas=tuple(config["cv_alphas"]), folds=config["cv_folds"], generator=gen,
            )
            cved[readout].append(acc)
            alphas[readout].append(chosen)
        print(f"  seed {seed} done  (surface {surface[-1]:.3f})", flush=True)

    results = {
        "n_layers": int(fixed[config["readouts"][0]][0].shape[0]),
        "surface_baseline": surface,
        "probe_fixed_ridge": {r: torch.stack(v).tolist() for r, v in fixed.items()},
        "probe_cv": {r: torch.stack(v).tolist() for r, v in cved.items()},
        "cv_alpha": {r: torch.stack(v).tolist() for r, v in alphas.items()},
    }
    path = save_results(out_dir, manifest.finish(), results)
    return path, results
