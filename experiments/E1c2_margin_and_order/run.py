"""E1c-2 — fix E1c's two defects: degenerate label, and an untested prompt artefact.

WHY THIS EXPERIMENT EXISTS

E1c produced one clean positive and one clean negative:
  + the tile glyphs are affectively neutral (move bias +0.017)
  - the untrained policy is near-constant ("left" 92.4%, "right" 0.0%), so a
    binary toward/not-toward label is 7.5/92.5 imbalanced and the move probe
    scored 0.66 against a 0.925 majority-class baseline -- worse than trivial

Two fixes, both cheap:

  1. CONTINUOUS TARGET. Replace the binary argmax label with the logit margin
     logit(toward) - logsumexp(other three). It has variance even when the argmax
     never changes, and R^2 has a meaningful zero (predicting the mean), so the
     majority-class pathology cannot recur.

  2. WORD-ORDER CONTROL. The prompt lists "up, down, left, right". A 92% "left"
     rate might be a lexical prior of the model or an artefact of list position.
     Running a reversed and a rotated order separates them. This matters beyond
     E1c: if list position drives the policy, every organism inherits the artefact
     and the maze prompt needs redesigning before any RL is spent.

PRE-COMMITMENTS, fixed before the run:
  - If "left" stays dominant under permutation, the prior is lexical -- a property
    of the model worth reporting, and harmless to the design.
  - If dominance follows a POSITION instead (e.g. always the third-listed word),
    the prompt is the cause. That would invalidate E1c's move statistics and
    require a prompt change before organisms are built.
  - Margin R^2 is expected to be modestly positive: the model's move preference
    does depend on the grid somewhat, even if the argmax rarely changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from calibration.analysis import (  # noqa: E402
    direction_alignment,
    probe_direction,
    probe_direction_from_labels,
    probe_r2,
)
from calibration.capture import MOVE_WORDS, move_logits, pooled_resid  # noqa: E402
from calibration.maze import TILE_GOLD, TILE_MOLD, state_bank  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

ORDERS = {
    "canonical": ("up", "down", "left", "right"),
    "reversed": ("right", "left", "down", "up"),
    "rotated": ("left", "right", "up", "down"),
}

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "n_states": 96,
    "batch_size": 8,
    "seeds": [0, 1, 2],
    "direction": [-1, 0],
    "toward_move": "up",
    "readout": "mean_grid",
    "ridge": 1.0,
    "orders": list(ORDERS),
}


def _margin(logits, toward_idx):
    """logit(toward) - logsumexp(others). Continuous, unbounded, zero-centred-ish."""
    other = torch.cat([logits[:, :toward_idx], logits[:, toward_idx + 1 :]], dim=1)
    return logits[:, toward_idx] - other.logsumexp(dim=1)


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = out_dir or (Path(__file__).parent / "results")
    manifest = RunManifest(
        experiment="E1c2_margin_and_order",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes="Untrained model. Continuous logit margin + move-word order control.",
    )
    toward = MOVE_WORDS.index(config["toward_move"])

    order_dist = {o: [] for o in config["orders"]}
    margin_gap, margin_r2, alignment = [], [], []

    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        direction = tuple(config["direction"])
        g_mold = state_bank(TILE_MOLD, n=config["n_states"], seed=seed, direction=direction)
        g_gold = state_bank(TILE_GOLD, n=config["n_states"], seed=seed, direction=direction)

        # --- word-order control ---
        for name in config["orders"]:
            lg, _ = move_logits(
                g_mold, model, tokenizer,
                batch_size=config["batch_size"], move_order=ORDERS[name],
            )
            counts = torch.bincount(lg.argmax(-1), minlength=len(MOVE_WORDS)).float()
            order_dist[name].append((counts / counts.sum()).tolist())

        # --- continuous margin, canonical order ---
        lg_mold, _ = move_logits(g_mold, model, tokenizer, batch_size=config["batch_size"])
        lg_gold, _ = move_logits(g_gold, model, tokenizer, batch_size=config["batch_size"])
        m_mold, m_gold = _margin(lg_mold, toward), _margin(lg_gold, toward)
        margin_gap.append(float(m_gold.mean() - m_mold.mean()))

        h_mold = pooled_resid(g_mold, model, tokenizer, batch_size=config["batch_size"])[config["readout"]]
        h_gold = pooled_resid(g_gold, model, tokenizer, batch_size=config["batch_size"])[config["readout"]]
        h_all = torch.cat([h_mold, h_gold])
        margins = torch.cat([m_mold, m_gold])

        margin_r2.append(probe_r2(h_all, margins, ridge=config["ridge"], generator=gen))

        w_tile = probe_direction(h_mold, h_gold, ridge=config["ridge"])
        w_move = probe_direction_from_labels(h_all, margins, ridge=config["ridge"])
        alignment.append(direction_alignment(w_tile, w_move))

        print(
            f"  seed {seed}: margin gap {margin_gap[-1]:+.4f}  "
            f"max R2 {float(margin_r2[-1].max()):.3f}  "
            f"max align {float(alignment[-1][1:].max()):.3f}",
            flush=True,
        )

    results = {
        "n_layers": int(alignment[0].shape[0]),
        "move_words": list(MOVE_WORDS),
        "orders": {k: list(v) for k, v in ORDERS.items()},
        "order_move_dist": order_dist,
        "margin_gap": margin_gap,
        "margin_r2": torch.stack(margin_r2).tolist(),
        "alignment": torch.stack(alignment).tolist(),
    }
    path = save_results(out_dir, manifest.finish(), results)
    return path, results
