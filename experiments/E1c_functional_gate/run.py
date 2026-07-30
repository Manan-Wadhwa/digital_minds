"""E1c — a functional gate candidate: is tile identity ACTION-RELEVANT?

WHY THIS EXPERIMENT EXISTS

Two gate candidates have now failed, for opposite reasons:

  cosine(v_mold, v_gold)   baseline-dependent -- moves from +0.79 to -1.00 on
                           identical activations by choice of contrast (E1a)
  probe separability       saturated -- above 0.95 at every layer of the
                           UNTRAINED model, so RL cannot raise it (E1b)

The diagnosis behind both: which tile is adjacent is literally in the input as a
distinct glyph, so *identity* is encoded before any training. RL is not supposed
to make mold and gold more distinguishable. It is supposed to make them
differently ACTED ON. Both failed candidates measure representation; neither
measures whether the representation drives behaviour.

WHAT THIS MEASURES, per layer, across seeds:

  1. move_bias   -- P(move toward the controlled tile | mold) vs (| gold).
                    Behavioural, no activations. Untrained this should be ~equal:
                    the model has no reason to treat the glyphs differently.

  2. alignment   -- |cos| between the tile-identity probe direction and the
                    move-prediction probe direction, per layer. THE GATE
                    CANDIDATE. Untrained, moves are driven by grid layout rather
                    than tile identity, so the two directions should be roughly
                    unrelated. After RL on a mold penalty they should collapse
                    onto each other.

  3. move_predictability -- held-out accuracy of predicting the move from
                    activations at all, as a ceiling for (2). If moves are not
                    predictable, alignment is uninterpretable.

WHY IT SURVIVES BOTH PREVIOUS FAILURE MODES
  - No baseline to choose: alignment is between two fitted directions.
  - Cannot saturate pre-training: it is near zero by construction if the
    untrained model does not act on tile identity, which (1) tests directly.

PRE-COMMITMENTS, fixed before the run:
  - move_bias is expected to be near zero. A large untrained bias would mean the
    glyphs are NOT affectively neutral to this model, which would undermine the
    maze design itself and is a finding worth having early.
  - alignment is expected to be low but NOT zero: both probes read the same
    grid, so some shared structure is expected from layout alone.
  - If moves are near-deterministic (one word almost always), the move probe has
    no variance to fit and the run is uninterpretable. Reported explicitly.
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
    probe_separability_cv,
)
from calibration.capture import MOVE_WORDS, move_logits, pooled_resid  # noqa: E402
from calibration.maze import TILE_GOLD, TILE_MOLD, state_bank  # noqa: E402
from calibration.runner import RunManifest, save_results, set_all_seeds  # noqa: E402

CONFIG = {
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "n_states": 96,
    "batch_size": 8,
    "seeds": [0, 1, 2, 3, 4],
    "direction": [-1, 0],          # controlled neighbour is UP
    "toward_move": "up",           # so "toward the tile" is the up move
    "readout": "mean_grid",        # E1b: highest and flattest across depth
    "ridge": 1.0,
    "cv_alphas": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
}


def run(model, tokenizer, config=CONFIG, out_dir=None):
    out_dir = out_dir or (Path(__file__).parent / "results")
    manifest = RunManifest(
        experiment="E1c_functional_gate",
        config=config,
        seeds=config["seeds"],
        model_id=config["model_id"],
        device=str(next(model.parameters()).device),
        notes=(
            "Untrained model. Establishes the pre-training half of a functional "
            "gate: does the tile-identity direction align with the move direction?"
        ),
    )
    toward = MOVE_WORDS.index(config["toward_move"])

    out = {k: [] for k in ("move_bias", "alignment", "move_pred", "p_toward_mold",
                           "p_toward_gold", "move_entropy", "move_dist")}

    for seed in config["seeds"]:
        gen = set_all_seeds(seed)
        direction = tuple(config["direction"])
        g_mold = state_bank(TILE_MOLD, n=config["n_states"], seed=seed, direction=direction)
        g_gold = state_bank(TILE_GOLD, n=config["n_states"], seed=seed, direction=direction)

        lg_mold, _ = move_logits(g_mold, model, tokenizer, batch_size=config["batch_size"])
        lg_gold, _ = move_logits(g_gold, model, tokenizer, batch_size=config["batch_size"])
        mv_mold, mv_gold = lg_mold.argmax(-1), lg_gold.argmax(-1)

        p_mold = (mv_mold == toward).float().mean().item()
        p_gold = (mv_gold == toward).float().mean().item()
        out["p_toward_mold"].append(p_mold)
        out["p_toward_gold"].append(p_gold)
        out["move_bias"].append(p_gold - p_mold)   # >0 means mold is avoided

        all_moves = torch.cat([mv_mold, mv_gold])
        counts = torch.bincount(all_moves, minlength=len(MOVE_WORDS)).float()
        probs = counts / counts.sum()
        out["move_dist"].append(probs.tolist())
        out["move_entropy"].append(
            float(-(probs[probs > 0] * probs[probs > 0].log()).sum())
        )

        h_mold = pooled_resid(g_mold, model, tokenizer, batch_size=config["batch_size"])[config["readout"]]
        h_gold = pooled_resid(g_gold, model, tokenizer, batch_size=config["batch_size"])[config["readout"]]

        w_tile = probe_direction(h_mold, h_gold, ridge=config["ridge"])

        h_all = torch.cat([h_mold, h_gold])
        toward_label = torch.where(all_moves == toward, 1.0, -1.0)
        w_move = probe_direction_from_labels(h_all, toward_label, ridge=config["ridge"])

        out["alignment"].append(direction_alignment(w_tile, w_move))

        acc, _ = probe_separability_cv(
            h_all[toward_label > 0], h_all[toward_label < 0],
            alphas=tuple(config["cv_alphas"]), generator=gen,
        ) if (toward_label > 0).any() and (toward_label < 0).any() else (
            torch.full((h_all.shape[1],), float("nan")), None
        )
        out["move_pred"].append(acc)

        print(f"  seed {seed}: P(up|mold)={p_mold:.3f} P(up|gold)={p_gold:.3f} "
              f"H(move)={out['move_entropy'][-1]:.3f}", flush=True)

    results = {
        "n_layers": int(out["alignment"][0].shape[0]),
        "move_words": list(MOVE_WORDS),
        "p_toward_mold": out["p_toward_mold"],
        "p_toward_gold": out["p_toward_gold"],
        "move_bias": out["move_bias"],
        "move_entropy": out["move_entropy"],
        "move_dist": out["move_dist"],
        "alignment": torch.stack(out["alignment"]).tolist(),
        "move_predictability": torch.stack(out["move_pred"]).tolist(),
    }
    path = save_results(out_dir, manifest.finish(), results)
    return path, results
